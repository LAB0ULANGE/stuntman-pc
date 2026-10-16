---
name: ps2-release
description: Prepare and publish a public release of a PS2 PC port made with this repository - what may and may not be in the public repository and the download, the legal checklist, the packaging script and its checks, versioning, release notes, and publishing on GitHub with the user's go-ahead. Use before any commit that might add game material, before a push, and when making a release.
---

# Releasing a port

The repository is public on GitHub. Every commit pushed is published - and its
history with it: something committed and deleted later is still in the
history. So what goes into a commit is checked **before** it is committed.

## Never in the repository

- Disc images (`*.iso`), anything extracted from a disc (`extracted/`), the
  disassembly (`asm/`), builds (`build/`), console BIOS dumps (`bios/`).
- **The game's code in any form**: lifted, derived or hand-reconstructed C++
  (`games/<name>/src/`, `games/<name>/include/<game>/` - all ignored), VU
  microcode, anything generated from the executable that carries its code.
  The tools regenerate it from the player's own disc.
- **The game's assets**: textures, models, sounds, music, movies, fonts, text
  dumps, its logo or any art taken from it - also not inside an icon, a
  screenshot in the docs, a test fixture or a hex array in a source file.
- Personal data: the user's paths (`C:\Users\...`), keys, tokens.

`.gitignore` covers the paths; it does not cover content. Before a commit,
`git diff --cached --name-only`, and for anything new and binary or large, look
at what it is. What *may* be committed: the toolchain and the port's code, the
project's own art (drawn by scripts in the repo), documentation of how the
game works (addresses, names, findings), checksums of test pictures.

If something did get committed and has not been pushed: remove it from the
history (`git filter-branch --index-filter 'git rm -r --cached
--ignore-unmatch <paths>' -- --all`, then drop `refs/original`, expire the
reflog, `git gc --prune=now`) after a backup (`git bundle create <file> --all`
outside the repository). Once pushed, it is public: tell the user at once -
rewriting published history is their decision.

## The legal position (say it plainly to the user)

- The repository holds no game assets and none of the game's code.
- The release executable **does** contain the game's program, translated to
  C++ and compiled - with no assets: it reads everything from the player's own
  disc image and refuses one that is not the game's. That is the grey area the
  public decompilation projects are in; it is the user's decision to publish
  it, not a thing to hide or to claim is "legal".
- The README and the release say: unofficial, not affiliated with the game's
  publisher or developer, bring your own disc (the exact release, e.g.
  NTSC-U SLUS-21050), no game content included; the trademark is its owner's.
- The project's own code is under the user's chosen licence (MIT here:
  `LICENSE`, which says it does not cover the game). The libraries built in
  (SDL 3, volk, glslang, Vulkan-Headers) ship their licence texts with the
  binary (THIRD-PARTY-NOTICES.txt).

## Making a release

1. **Version** in `games/<name>/game.json` (`"version": "0.1.0-alpha"`): the
   executable's version information and the log's first line.
2. **Build** (`python engine/tools/port.py --game games/<name> build`) with
   the release setup the port's CMake already has: the C/C++ runtime linked in
   (`/MT` - nothing for the player to install), the symbols' file named
   without a path (`/PDBALTPATH:%_PDB%`), source paths trimmed to the
   repository (`/d1trimfile`), nothing read at run time from where it was
   built (data built in).
3. **The icon** is the project's own (`port/icon.ico`). A logo icon a player
   makes from their disc (`build/port/icon.ico`) is for their own build only.
4. **Regressions** pass (skill `ps2-port`) and the benchmark has no new late
   frames.
5. **Package**: `python games/<name>/port/tools/package.py` (the game's own
   until a second game releases: then copy Burnout 3's
   (burnout3-pc), and the second release moves the common part into
   `engine/tools/`) writes
   `build/release/<Name>-PC-<version>-windows-x64.zip` - the executable, the
   player's README (`port/release/README.txt`: what is needed, getting started,
   controls, settings, the files beside the executable, known issues, the
   notice) and the third-party licences. It refuses a build with the disc-made
   logo, one that imports the Visual C++ runtime, or one that names a folder of
   the build machine. Don't bypass a refusal: fix its cause.
6. **Test it as a player would**: unzip into an empty folder, start it with no
   arguments (a `pc_settings.ini` with `disc=` beside it, for an unattended
   check: the picker otherwise waits for a person), check the log (version,
   disc, boot, sound) and that it writes only beside itself.
7. **Ask the user** before anything public: pushing, creating or changing a
   release, making a repository public. Show what will be published.
8. **Publish** (with their go-ahead): `git push`, then
   `gh release create v<version> <zip> --title ... --notes-file ... --prerelease`
   for an alpha/beta; download the asset again and compare its SHA-256 with
   the tested zip.

## Release notes

What it is (one line, unofficial, bring your own disc - which release),
what it does, getting started in three steps, what it needs (Windows
version, AVX2, Vulkan 1.3), known issues, how to report a problem (the log
beside the executable). No promises about future versions; no game
screenshots made from someone's disc unless the user decides to.
