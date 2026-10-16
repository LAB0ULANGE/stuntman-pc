/*
 * A VIF's command stream (EE User's Manual, chapter 6): VIFcodes and their
 * data, delivered to the VU behind the VIF and to the GS.
 *
 * MPG loads microprograms into the VU's instruction memory, UNPACK expands
 * packed vectors into its data memory (formats S/V2/V3/V4 of 32, 16 or 8 bits
 * and V4-5, with STMASK/STROW/STCOL masking, STMOD's offset and difference
 * modes, and STCYCL's skipping and filling writes), and MSCAL, MSCALF and MSCNT
 * run the microprogram - to its end, at once: the EE sees it done by the time
 * it could look. VIF1's double buffering (BASE, OFFSET, TOPS and the FLG bit of
 * UNPACK) and ITOP feed XTOP and XITOP. DIRECT and DIRECTHL carry GIF data on
 * PATH2; MSKPATH3 masks PATH3, which code that hands PATH3 over waits on.
 */

#ifndef PS2HOST_VIF_H
#define PS2HOST_VIF_H

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ps2host {

class Vu;

class VifStream {
public:
    /* The VU behind this VIF (VU1 for VIF1); none drops VU traffic. */
    Vu* vu = nullptr;
    bool double_buffered = false;   /* VIF1's BASE/OFFSET/TOPS */
    /* Called when MSKPATH3 masks (true) or unmasks (false) PATH3. */
    std::function<void(bool masked)> on_mask_path3;
    /* Each word of DIRECT/DIRECTHL data: GIF data on PATH2. */
    std::function<void(std::uint32_t word)> on_direct;
    /* DIRECT's data a run at a time, from guest address `from` (Words); when
       unset, a word at a time through on_direct. */
    std::function<void(const std::uint32_t* words, std::uint32_t count, std::uint32_t from)> on_direct_block;
    /* Commands worth seeing in a trace (FLUSHA), when set. */
    std::function<void(const char* what)> on_trace;

    /* The next 32-bit word of the stream, from the FIFO or a DMA transfer. */
    void Word(std::uint32_t word);
    /* `count` words from guest address `from`, as Word each (keeping
       g_dma_source), DIRECT's data handed on a run at a time. */
    void Words(const std::uint32_t* words, std::uint32_t count, std::uint32_t from);

    std::uint64_t commands() const { return commands_; }
    std::uint64_t data_words() const { return data_words_; }
    /* For the report: how often each command came, by name. */
    std::string Summary() const;

    // Registers the EE can read (VIF1_TOP, VIF1_TOPS, ITOP, ...).
    std::uint32_t top = 0, tops = 0, base = 0, offset = 0, itop = 0, itops = 0, dbf = 0;
    std::uint32_t mask = 0, mode = 0, row[4] = {}, col[4] = {};

private:
    enum class Data { None, Direct, Mpg, Unpack, Mask, Row, Col };
    std::uint32_t pending_ = 0;     /* data words still owed to the last command */
    Data data_ = Data::None;
    std::uint32_t cycle_length_ = 0, write_length_ = 0;     /* STCYCL's CL and WL */
    std::uint32_t data_index_ = 0;  /* words of the current command's data so far */
    std::uint32_t mpg_address_ = 0;
    std::uint32_t end_pc_ = 0;      /* where MSCNT continues */
    struct Unpack {
        std::uint32_t address = 0, format = 0, num = 0, written = 0;
        bool masked = false, unsigned_ = false;
        std::vector<std::uint8_t> bytes;
    } unpack_;
    std::uint64_t commands_ = 0, data_words_ = 0;
    std::array<std::uint64_t, 128> by_command_{};

    /* Words of data an UNPACK of `num` elements in format `format` reads. */
    std::uint32_t UnpackWords(std::uint32_t format, std::uint32_t num) const;
    void UnpackData(std::uint32_t word);
    /* A run of UNPACK's data words: whole elements straight from them while
       none is left part-read, the rest a word at a time. */
    void UnpackRun(const std::uint32_t* words, std::uint32_t count);
    /* One element's components from its bytes, as UNPACK expands them. */
    void Element(const std::uint8_t* b, std::uint32_t c[4]) const;
    void UnpackFill();
    void WriteElement(const std::uint32_t value[4], bool from_data);
    void Start(std::uint32_t pc);
};

/* The VIF1 stream fed by stores to its FIFO, 0x10005000. */
VifStream& Vif1();
/* VIF0's, at 0x10004000; it feeds VU0. */
VifStream& Vif0();

}  // namespace ps2host

#endif  /* PS2HOST_VIF_H */
