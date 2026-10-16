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

## COP0 Count: busy-waits and (probably) frame timing

Every `mfc0` in the executable (and in the runs copied from `.data`): 102. Only
Count (`$9`, 60 reads) is game code; the rest is the SDK and the kernel -
Status (20: the DI/EI pairs at `0x0025CCA0`-`0x0025D1B8`, `0x00264628`,
`EnableIntr` `0x00264670`, which test EIE and expect it clear on this host),
TagLo (cache operations, `0x0025CDF0`, `0x0025CF30`), the exception vectors at
`0x002651F8`-`0x00265448` (Cause, EPC, BadVAddr, BadPAddr), and TLB management
in the run copied to `0x80075000` (Index, PageMask, EntryHi/Lo, Wired).

Count is read by 14 functions:
- `0x001A3130` waits on it after reading from `STUNTMAN.IMG`:
  `s1 = Count; do { ... 0x0016C400 / 0x001D4FA8 ... } while (Count - s1 <= 0x960000)`
  at `0x001A3208` (about 33 ms at 294.912 MHz). With Count read as 0 it never
  ended - and, two instructions once compiled, it also lost the watchdog's
  single stop write (fixed: the watchdog writes it again every 10 ms).
- `0x002039D0`, `0x00203E08`, `0x00204010`, `0x002043C8`, `0x00204568`,
  `0x00204758`, `0x00204950`, `0x00204AA0`, `0x00204C08`, `0x00204D70`: ten
  neighbours reading it 4 to 6 times each - probably frame-time measurement.
  **If the game feeds this into its physics, it will matter for the cars' AI**:
  check what these values drive before tuning anything there. Under
  `--virtual-time` Count follows the virtual clock, so scripted runs stay
  reproducible.
- `0x001507C8`, `0x00160000`, `0x00182EC0`, `0x00205360`: one read each.

The port answers Count as the EE's cycles since the run began, from the host's
clock (`lifted::ReadCop0`, `engine/host/src/timers.cpp`); every other register
reads as 0 and is noted. Verification keeps 0 everywhere, so verdicts mean what
they meant. Listed with a scan of every word of `.text` and the copied runs.
