"""Render derived functions against recovered struct types.

The emitted code stays address-based - that is what lets it be compiled and run
against the interpreter - but a value whose type was recovered becomes a *view*:
a thin object over a guest address with one accessor per field. A pointer field's
accessor returns the view of the type it points to, so a chain of loads reads as
a chain of field accesses:

    ReadS32(ReadS32(arg0 + 4) + 8)      becomes      arg0.f_04().f_08()

A load is shown as a field access only when it matches the field exactly - same
offset, same width and signedness as the field's recovered read. Anything else
stays a plain Read* call on the view's `.address`, so a type that is wrong or
incomplete can make the source less tidy but never different in meaning; and the
compiled-code check runs every function in this form.
"""

from __future__ import annotations

from . import layout
from .paths import _split_address
from .structs import GLOBALS, Catalog, _offset_name
from .symexec import (Arg, Binary, Const, Load, address_of, arg_name, arg_renderer,
                      children,
                      load_renderer,
                      simplify)

READERS = {
    (1, True): "ReadS8", (1, False): "ReadU8",
    (2, True): "ReadS16", (2, False): "ReadU16",
    (4, True): "ReadS32", (4, False): "ReadU32",
    (8, False): "ReadU64",
}
WRITERS = {1: "Write8", 2: "Write16", 4: "Write32", 8: "Write64"}
VALUE_TYPES = {
    (1, True): "s8", (1, False): "u8", (2, True): "s16", (2, False): "u16",
    (4, True): "s32", (4, False): "u32", (8, False): "u64",
}


def _hex(value: int, signed: bool = False) -> str:
    text = "0x%X" % abs(value)
    if value < 0:
        return "-" + text
    return "+" + text if signed else text


MAX_STRIDE = 0x10000
# Global arrays live in the program's data (layout.image_start..image_end).
MASK64 = (1 << 64) - 1


def _sum_terms(expr):
    """The operands of a chain of additions."""
    if isinstance(expr, Binary) and expr.op == "add":
        yield from _sum_terms(expr.left)
        yield from _sum_terms(expr.right)
    else:
        yield expr


def _combine(a, b, sign):
    """a + sign*b for linear forms (coefficient, atom, constant); None if the
    atoms differ."""
    if a[1] is not None and b[1] is not None and a[1] != b[1]:
        return None
    return ((a[0] + sign * b[0]) & MASK64, a[1] if a[1] is not None else b[1],
            (a[2] + sign * b[2]) & MASK64)


def linear(expr):
    """expr as (coefficient, atom, constant): coefficient*atom + constant, mod 2^64.

    The atom is the one non-constant subexpression; anything that is not a sum,
    difference, constant shift or constant multiple of it is itself the atom.
    Returns None when two different atoms appear.
    """
    if isinstance(expr, Const):
        return (0, None, expr.value & MASK64)
    if isinstance(expr, Binary):
        if expr.op in ("add", "sub"):
            left, right = linear(expr.left), linear(expr.right)
            if left is None or right is None:
                return None
            return _combine(left, right, 1 if expr.op == "add" else -1)
        if expr.op in ("shl", "dshl") and isinstance(expr.right, Const):
            inner = linear(expr.left)
            if inner is None:
                return None
            factor = 1 << (expr.right.value & (31 if expr.op == "shl" else 63))
            return ((inner[0] * factor) & MASK64, inner[1], (inner[2] * factor) & MASK64)
        if expr.op == "mul" and isinstance(expr.right, Const) or (
                expr.op == "mul" and isinstance(expr.left, Const)):
            constant, other = ((expr.right, expr.left) if isinstance(expr.right, Const)
                               else (expr.left, expr.right))
            inner = linear(other)
            if inner is None:
                return None
            factor = constant.value
            return ((inner[0] * factor) & MASK64, inner[1], (inner[2] * factor) & MASK64)
    return (1, expr, 0)


def global_name(address: int) -> str:
    return "g_%08X" % (address & 0xFFFFFFFF)


def object_name(address: int) -> str:
    """A global object's name. Distinct from the scalar at the same address: an
    object's first field and the object itself share an address."""
    return "g_%08X_obj" % (address & 0xFFFFFFFF)


def embedded_accessor(offset: int) -> str:
    return "e_%s" % _offset_name(offset)


def accessor(offset: int) -> str:
    return "f_%s" % _offset_name(offset)


def _signed_offset(offset: int) -> int:
    offset &= 0xFFFFFFFF
    return offset - (1 << 32) if offset & 0x80000000 else offset


class Typing:
    """Type information for one function's expressions."""

    def __init__(self, catalog: Catalog, function: int) -> None:
        self.catalog = catalog
        self.function = function
        self.used = set()
        self.used_globals = set()        # addresses of scalar globals read or written
        self.used_objects = set()        # (address, type) of global objects
        self.used_global_arrays = set()  # addresses of global arrays indexed

    def type_of(self, expr):
        """The recovered type of a value, as a catalog root, or None."""
        if isinstance(expr, Arg):
            return self.catalog.slot(self.function, expr.index)
        if isinstance(expr, Load):
            access = self.access(expr)
            if access is not None:
                _base, _path, owner, local, _index = access
                return self.catalog.pointee(owner, local)
        return None

    def _candidates(self, address):
        """Each way to reach a typed address, best first.

        Yields (base, path, owner, local, field, index). `index` is None for a plain
        field; an int for an element of a run of fields the code walked with an
        unrolled loop; or an expression when the address is indexed by a variable,
        in which case `local` is element 0 of the array.
        """
        indexed = self.decompose(address)
        if indexed is not None:
            base, root, offset, stride, index = indexed
            field = self.catalog.field(root, offset)
            if field is not None and self.catalog.arrays(root).get(offset, (None,))[0] == stride:
                yield base, [], root, offset, field, index
            return
        try:
            base, offset = _split_address(address)
        except Exception:
            return
        if base is None:
            # A constant address: a global variable, or a field of a global object.
            root = self.catalog.globals
        else:
            root = self.type_of(base)
        if root is None:
            return
        offset = _signed_offset(offset)
        element = self.catalog.element(root, offset)
        if element is not None:
            start, index = element
            yield base, [], root, start, self.catalog.field(root, start), index
        for path, owner, local, field in self.catalog.candidates(root, offset):
            yield base, path, owner, local, field, None

    def decompose(self, address):
        """(base, root, offset, stride, index) for an address indexed by a variable.

        The address must be exactly base + offset + stride * index, with the base a
        typed value (or absent, for a global array). Shifts, multiplies, additions
        and subtractions by constants are exact in the 64-bit model, so
        `arg0 + 0x10 + ((i << 2) + i << 4)` is arg0 + 0x10 + 0x50 * i.
        """
        terms = list(_sum_terms(address_of(address)))
        base, root = None, self.catalog.globals
        rest = []
        for term in terms:
            typed_root = None if isinstance(term, Const) else self.type_of(term)
            if typed_root is not None and base is None:
                base, root = term, typed_root
            else:
                rest.append(term)
        if root is None:
            return None
        form = (0, None, 0)
        for term in rest:
            part = linear(term)
            if part is None:
                return None
            form = _combine(form, part, 1)
            if form is None:
                return None
        coefficient, index, constant = form
        if index is None or not 1 <= coefficient <= MAX_STRIDE:
            return None
        if base is None and not (layout.CURRENT.image_start <= constant & 0xFFFFFFFF
                                 < layout.CURRENT.image_end):
            # A "global array" at 0x20 indexed by a pointer is exact arithmetic but
            # a false description: the pointer is the base, not an index.
            return None
        return base, root, _signed_offset(constant), coefficient, index

    def access(self, load):
        """(base, embed path, owner, offset, index) when a load is exactly a field read."""
        for base, path, owner, local, field, index in self._candidates(load.address):
            if self.catalog.read_flavour(field) == (load.size, load.signed):
                return base, path, owner, local, index
        return None

    def store_target(self, store):
        """(base, embed path, owner, offset, index) when a store is exactly a field write."""
        for base, path, owner, local, field, index in self._candidates(store.address):
            if self.catalog.write_width(field) == store.size:
                return base, path, owner, local, index
        return None

    def object(self, base, path) -> str:
        """The view a field belongs to: the base's view, then any embedded parts.

        With no base the field is global: the first embedded part is then a global
        object, named for its address.
        """
        if base is None:
            at, inner = path[0]
            self.used.add(inner)
            self.used_objects.add((at, inner))
            text, path = object_name(at), path[1:]
        else:
            text = self.view(base)
        for at, inner in path:
            self.used.add(inner)
            text += ".%s()" % embedded_accessor(at)
        return text

    def index_text(self, index) -> str:
        """An element number or index expression, rendered with this typing.

        Installs the hooks itself: it is also called from statement-level code,
        outside any expression rendering, and the default renderer would print a
        typed argument without `.address`.
        """
        if index is None:
            return ""
        if isinstance(index, int):
            return str(index)
        with load_renderer(self.render_load), arg_renderer(self.render_arg):
            return simplify(index).render()

    def read(self, base, path, local, index=None) -> str:
        """The expression reading a located field, or an element of one."""
        arguments = self.index_text(index)
        if base is None and not path:
            (self.used_global_arrays if index is not None else self.used_globals).add(local)
            return "%s(%s)" % (global_name(local), arguments)
        return "%s.%s(%s)" % (self.object(base, path), accessor(local), arguments)

    def write(self, holder, base, path, local, value, index=None) -> str:
        """The statement writing a located field; `holder` is a captured view and
        `index` an element number, an index expression, or text already captured."""
        arguments = value if index is None else "%s, %s" % (
            index if isinstance(index, str) else self.index_text(index), value)
        if base is None and not path:
            (self.used_global_arrays if index is not None else self.used_globals).add(local)
            return "set_%s(%s);" % (global_name(local), arguments)
        return "%s.set_%s(%s);" % (holder or self.object(base, path), accessor(local),
                                   arguments)

    @staticmethod
    def needs_capture(base) -> bool:
        """Whether the view a store goes through reads memory to compute."""
        return base is not None and not isinstance(base, Arg)

    def view(self, expr) -> str:
        """Render a typed value as its view object."""
        root = self.type_of(expr)
        assert root is not None
        self.used.add(root)
        if isinstance(expr, Arg):
            return arg_name(expr.index)
        base, path, owner, local, index = self.access(expr)
        self.used.add(owner)
        return self.read(base, path, local, index)

    def type_name(self, root) -> str:
        return self.catalog.names[root]

    # -- hooks for the symexec renderer ------------------------------------

    def render_arg(self, node) -> str:
        if self.type_of(node) is not None:
            self.used.add(self.type_of(node))
            return "%s.address" % arg_name(node.index)
        return arg_name(node.index)

    def render_load(self, node, render_child) -> str:
        access = self.access(node)
        if access is not None:
            base, path, owner, offset, index = access
            self.used.add(owner)
            text = self.read(base, path, offset, index)
            if self.type_of(node) is not None:
                self.used.add(self.type_of(node))
                return text + ".address"
            return text
        reader = READERS.get((node.size, node.signed))
        if reader is None:
            from .symexec import Unsupported

            raise Unsupported("no reader for %d-byte load" % node.size)
        return "%s(%s)" % (reader, render_child(node.address))


def header(catalog: Catalog, roots, source_name: str, globals_used=(), objects=(),
           global_arrays=()) -> str:
    """The view classes for the given types, and the globals used, as a C++ header."""
    roots = sorted((root for root in roots if root != catalog.globals),
                   key=lambda root: catalog.names[root])
    included = set(roots)
    lines = [
        "/*",
        " * Recovered struct types - GENERATED, do not edit by hand.",
        " *",
        " * Produced by engine/tools/derive.py from %s, using the type recovery in" % source_name,
        " * tools/ps2decomp/structs.py. Each type was inferred from how the original code",
        " * uses pointers: the offsets and widths it accesses, and the arguments, return",
        " * values and pointer fields that carry the same object between functions.",
        " *",
        " * A type is a *view* over a guest address rather than a C++ struct with the",
        " * original layout, because the derived code runs against an emulated address",
        " * space. Each accessor reads or writes one field; a pointer field returns the",
        " * view of the type it points to. Only types the derived functions use are",
        " * listed, each with every field the recovery found, including fields no",
        " * derived function touches yet.",
        " *",
        " * The names are placeholders derived from the evidence: S_0035F0A0_a0 is the",
        " * type of argument 0 of func_0035F0A0, and S_..._p1C is what its field 0x1C",
        " * points to. Global variables are named by address: g_004876E8() reads the",
        " * one at 0x4876E8, S_g004876E8 is what it points to, and g_0049A278_obj is a",
        " * global object of a recovered type.",
        " */",
        "",
        "#ifndef B3_GENERATED_TYPES_H",
        "#define B3_GENERATED_TYPES_H",
        "",
        '#include "ps2rt/generated.h"',
        "",
        "namespace generated {",
        "",
        "/* Every view is a guest address; this is the shared base. */",
        "struct View {",
        "    u64 address;",
        "    View(u64 value = 0) : address(value) {}",
        "};",
        "",
    ]
    for root in roots:
        lines.append("struct %s;" % catalog.names[root])
    lines.append("")

    definitions = []
    for root in roots:
        name = catalog.names[root]
        fields = catalog.roots[root]
        offsets = sorted(fields)
        lines.append("/* %d function(s); %d field(s) at %s..%s */"
                     % (len(catalog.functions(root)), len(fields),
                        _hex(offsets[0]), _hex(offsets[-1])))
        lines.append("struct %s : View {" % name)
        lines.append("    using View::View;")
        for offset in offsets:
            field = fields[offset]
            flavour = catalog.read_flavour(field)
            width = catalog.write_width(field)
            target = catalog.pointee(root, offset)
            kind = catalog.kind(field)
            note = "%d function(s)" % len(field.functions)
            displacement = ("address + 0x%X" % offset if offset > 0 else
                            "address - 0x%X" % -offset if offset < 0 else "address")
            if flavour is not None and flavour in READERS:
                if target is not None and target in included:
                    returns = catalog.names[target]
                    describe = "-> %s" % returns
                else:
                    returns = "u64"
                    describe = "f32 bits" if kind == "float" else VALUE_TYPES[flavour]
                    if target is not None:
                        describe = "pointer to %s" % catalog.names[target]
                lines.append("    %s %s() const;  /* %s  %s, %s */"
                             % (returns, accessor(offset), _hex(offset, signed=True),
                                describe, note))
                definitions.append("inline %s %s::%s() const { return %s(%s); }"
                                   % (returns, name, accessor(offset), READERS[flavour],
                                      displacement))
            if width is not None and width in WRITERS:
                lines.append("    void set_%s(u64 value) const;" % accessor(offset))
                definitions.append(
                    "inline void %s::set_%s(u64 value) const { %s(%s, value); }"
                    % (name, accessor(offset), WRITERS[width], displacement)
                )
        for start, (stride, count) in sorted(catalog.arrays(root).items()):
            field = fields[start]
            flavour = catalog.read_flavour(field)
            width = catalog.write_width(field)
            target = catalog.pointee(root, start)
            element = ("address + 0x%X + index * 0x%X" % (start, stride) if start >= 0
                       else "address - 0x%X + index * 0x%X" % (-start, stride))
            extent = "%d elements" % count if count is not None else "length unknown"
            if flavour in READERS:
                returns = (catalog.names[target] if target is not None and target in included
                           else "u64")
                lines.append("    %s %s(u64 index) const;  /* array: every 0x%X bytes, %s */"
                             % (returns, accessor(start), stride, extent))
                definitions.append("inline %s %s::%s(u64 index) const { return %s(%s); }"
                                   % (returns, name, accessor(start), READERS[flavour], element))
            if width in WRITERS:
                lines.append("    void set_%s(u64 index, u64 value) const;" % accessor(start))
                definitions.append(
                    "inline void %s::set_%s(u64 index, u64 value) const { %s(%s, value); }"
                    % (name, accessor(start), WRITERS[width], element))
        for at, inner in sorted(catalog.embedded(root).items()):
            if inner not in included:
                continue
            lines.append("    %s %s() const;  /* %s  embedded %s */"
                         % (catalog.names[inner], embedded_accessor(at), _hex(at, signed=True),
                            catalog.names[inner]))
            definitions.append("inline %s %s::%s() const { return %s(address + 0x%X); }"
                               % (catalog.names[inner], name, embedded_accessor(at),
                                  catalog.names[inner], at))
        lines.append("};")
        lines.append("")
    lines.append("/* Accessor definitions, after every type is complete. */")
    lines.extend(definitions)
    lines += global_section(catalog, included, globals_used, objects, global_arrays)
    lines += ["", "}  // namespace generated", "", "#endif  /* B3_GENERATED_TYPES_H */", ""]
    return "\n".join(lines)


def global_section(catalog, included, globals_used, objects, global_arrays=()) -> list:
    """Accessors for the global variables and global objects the code uses.

    A scalar global is a pair of functions, `g_004876E8()` and `set_g_004876E8(v)`,
    reading at the width and signedness the code uses it with; one that holds a
    pointer returns the view of what it points to. A global object - one whose
    address is handed to functions that know its type - is a constant view.
    """
    lines = ["", "/* Global variables, by address. */"]
    fields = catalog.roots.get(catalog.globals, {})
    for address in sorted(globals_used):
        field = fields[address]
        flavour = catalog.read_flavour(field)
        width = catalog.write_width(field)
        target = catalog.pointee(catalog.globals, address)
        name = global_name(address)
        location = "0x%X" % (address & 0xFFFFFFFF)
        note = "%d function(s)" % len(field.functions)
        if flavour in READERS:
            if target is not None and target in included:
                returns, describe = catalog.names[target], "-> " + catalog.names[target]
            else:
                returns = "u64"
                describe = "f32 bits" if catalog.kind(field) == "float" else VALUE_TYPES[flavour]
            lines.append("/* %s  %s, %s */" % (location, describe, note))
            lines.append("inline %s %s() { return %s(%s); }"
                         % (returns, name, READERS[flavour], location))
        if width in WRITERS:
            if flavour not in READERS:
                lines.append("/* %s  written only, %s */" % (location, note))
            lines.append("inline void set_%s(u64 value) { %s(%s, value); }"
                         % (name, WRITERS[width], location))
    if global_arrays:
        lines += ["", "/* Global arrays, by address of element 0. */"]
        arrays = catalog.arrays(catalog.globals)
        for address in sorted(global_arrays):
            field = fields[address]
            stride = arrays[address][0]
            flavour = catalog.read_flavour(field)
            width = catalog.write_width(field)
            target = catalog.pointee(catalog.globals, address)
            element = "0x%X + index * 0x%X" % (address & 0xFFFFFFFF, stride)
            lines.append("/* 0x%X  every 0x%X bytes, %d function(s) */"
                         % (address & 0xFFFFFFFF, stride, len(field.functions)))
            if flavour in READERS:
                returns = (catalog.names[target] if target is not None and target in included
                           else "u64")
                lines.append("inline %s %s(u64 index) { return %s(%s); }"
                             % (returns, global_name(address), READERS[flavour], element))
            if width in WRITERS:
                lines.append("inline void set_%s(u64 index, u64 value) { %s(%s, value); }"
                             % (global_name(address), WRITERS[width], element))
    if objects:
        lines += ["", "/* Global objects, by address. */"]
        for address, inner in sorted(objects):
            lines.append("inline const %s %s{0x%X};"
                         % (catalog.names[inner], object_name(address), address & 0xFFFFFFFF))
    return lines


def _walk(expr):
    yield expr
    for child in children(expr):
        yield from _walk(child)


def _accesses(derivation):
    """Every (address, size, signed, is_store) a derivation performs."""
    seen = set()
    for path in derivation.paths:
        roots = list(path.conditions)
        for store in path.stores:
            roots += [store.address, store.value]
        if path.expression is not None:
            roots.append(path.expression)
        for root in roots:
            for node in _walk(root):
                if isinstance(node, Load) and id(node) not in seen:
                    seen.add(id(node))
                    yield node.address, node.size, node.signed, False
        for store in path.stores:
            yield store.address, store.size, None, True


def _untyped_indexed(typing, func, address, graph):
    """Like Typing.decompose, for an indexed access through an argument with no
    type yet - which then gets one. None if the address is not of that shape."""
    terms = list(_sum_terms(address_of(address)))
    for term in terms:
        if isinstance(term, Arg) and typing.type_of(term) is None:
            rest = [t for t in terms if t is not term]
            form = (0, None, 0)
            for other in rest:
                part = linear(other)
                form = None if part is None or form is None else _combine(form, part, 1)
            if form is None or form[1] is None or not 1 <= form[0] <= MAX_STRIDE:
                return None
            node = graph.node(("arg", func["addr"], term.index))
            return term, node, _signed_offset(form[2]), form[0], form[1]
    return None


def add_derived_evidence(graph, verified, rounds: int = 4) -> int:
    """Add the derived functions' own accesses to the type graph as evidence.

    The scan that builds the graph cannot follow a variable index, so an array of
    records walked by a loop contributes nothing. The derivations can: an unrolled
    loop appears as every concrete access it makes, and each one was verified
    against the original. So any access through a typed base at an offset the type
    does not yet have becomes a field, and an argument that has no type at all but
    is dereferenced - typically only inside an inlined callee - gets one.

    Repeated because each new field can type a pointer loaded from it.
    """
    added = 0
    for _round in range(rounds):
        catalog = Catalog(graph)
        before = added
        for func, derivation, _tested, _strong in verified:
            typing = Typing(catalog, func["addr"])
            for address, size, signed, is_store in _accesses(derivation):
                if size not in (1, 2, 4, 8):
                    continue
                indexed = typing.decompose(address)
                if indexed is None:
                    indexed = _untyped_indexed(typing, func, address, graph)
                if indexed is not None:
                    _base, root, offset, stride, _index = indexed
                    node = root
                    field = graph.field(node, offset)
                    if not field.strides:
                        added += 1
                    field.strides[stride] += 1
                    field.uses[(size, "derived", bool(signed) if not is_store else None,
                                is_store)] += 1
                    field.functions.add(func["addr"])
                    continue
                try:
                    base, offset = _split_address(address)
                except Exception:
                    continue
                offset = _signed_offset(offset)
                root = typing.type_of(base) if base is not None else catalog.globals
                if base is None and root is None:
                    node = graph.node(GLOBALS)
                elif root is None:
                    if not isinstance(base, Arg):
                        continue
                    node = graph.node(("arg", func["addr"], base.index))
                elif catalog.locate(root, offset) is not None:
                    continue
                else:
                    node = root
                field = graph.field(node, offset)
                if not field.uses:
                    added += 1
                # A derived load cannot tell `lwc1` from `lwu`, so it says nothing
                # about whether the field is a float.
                field.uses[(size, "derived", bool(signed) if not is_store else None,
                            is_store)] += 1
                field.functions.add(func["addr"])
        if added == before:
            break
    return added
