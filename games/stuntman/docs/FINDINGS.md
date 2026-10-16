# Stuntman: findings

What the binary revealed, as it is learned - the compiler (the ELF's
`.comment` section), middleware (RCS `$Id:` strings), the SDK version, the VU
regions, and then each subsystem the port had to understand (its menus, its
camera, its streaming, ...), with the addresses and how they were found.

## Boot: kernel extensions and the file I/O version check

- `0x0025D308` copies kernel extensions (a syscall table and alarms, 0x740
  bytes) from `.data` at `0x00279628` to `0x80076000` with `kCopy`, then
  installs them with `SetSyscall`; a second run is copied from `0x0027A660` to
  `0x80075000`. Both are translated at their run addresses (`port.py
  --discover`, now reading the source from whichever section holds it).
- The IOP is rebooted with `cdrom0:\IOPRP243.IMG;1`, whose library tags give
  `ioman   2430`, `fileio  2430`, `loadfile2430`.
- fio's initialisation (`0x002602E8`) binds server `0x80000001`, calls
  procedure `0xFF` and keeps the 4-byte answer at `0x002AEBE8`; `0x00260540`
  clears it after the reboot. Every open (`0x00260578`) first asks
  `0x002604B0`, which accepts only "2430" (`0x0027961C`) or the string at
  `*0x00279E78` ("....", `0x00294D28`); otherwise the open fails with
  `0xFFFEFFFC`. The first open is `cdrom0:\STUNTMAN.IMG;1` (magic "IMG3"), in
  a retry loop at `0x001A2D78` with `sceCdTrayReq(2)` (`0x001A3420`).
  Found with `--watch 002604B0 --watch 00252CC8 --watch-write 002AEBE8`.
