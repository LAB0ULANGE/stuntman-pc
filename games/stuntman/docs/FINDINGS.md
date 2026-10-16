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

## Asynchronous file reads: completion by SIF command 0x80000011

`0x001A3130` reads `STUNTMAN.IMG` asynchronously, then polls every 0x960000
cycles with `0x001A32A8`, which asks `0x002610E8(fd, 1, &state)` - fio's wait
for asynchronous I/O, mode 1 = no wait: PollSema on the semaphore whose id is
at `0x00279E74`, then a scan of a 32-word table at `0x00279DE8` (-1 = free; an
entry in use = a request in flight). The error path ("FileSystem: Disc error
during asynchronous read, retrying...", `0x0028B370`) was never taken: the
request simply never ended.

**Diagnostic corrected:** there is no `end_func`. fio's `read` (`0x00260BB8`)
calls `sceSifCallRpc(0x002AEBC0, 2, mode 0, request 0x002AD900 (0x20 bytes),
receive 0x002AE540 (4), end_func 0, 0)`. Its request is `+0` semaphore, `+4`
result address, `+8` 4, `+0x0C` IOP descriptor, `+0x10` buffer, `+0x14` size,
`+0x1C` slot; on a descriptor opened "no wait" (flag 0x8000) it puts the
semaphore id in the table and sends it negated. The end comes from the IOP's
fileio: a completion record written at the EE address the library gave at its
initialisation (procedure 0xFF's first word: `0x002AE580`), then **SIF command
0x80000011**, whose handler fio installs in `0x002602E8` with
`sceSifAddCmdHandler` (`0x0025E938`):
- record `+0` semaphore: >= 0 - copy the result (`+0x0C` bytes from `+0x10`)
  to `+0x08`, then `iSignalSema` (`0x0025C5D0`); < 0 - free the table entry
  holding `-semaphore`;
- record `+4` the procedure: a jump table at `0x00294CC0` copies per-procedure
  data (2 = read: the unaligned head and tail, sizes at `+0x14`, data at
  `+0x24`/`+0x64`, 0xA4 bytes in all; 11: 0x140 bytes; 12: 0x40; 23/25/26: up
  to 0x400).

**The handler starts at `0x0025FDD8`** (`lui $v0, 0x2B`), but the analysis
starts the function at `0x0025FDDC`, one instruction late: `port.py
--discover` translates it at its first call.

The host now sends that record and command for every asynchronous fileio
request (`engine/host/src/fileio.cpp`, `SendCmdToEe` in `iop.cpp`).

## The black screen: textures sent at VIF1's interrupt bit

The first screen drew ~78 primitives a frame, all black: its sprites sample
8-bit textures at GS 0x3540-0x3600 that were never uploaded (index 0 of an
alpha-ramp palette, RGB 0). The uploads are built every frame by `0x001CFFB0`
(via `0x001DB810` -> `0x001CFD48`, queued in a list at 0x0032xxxx with
`0x001CE798` / `0x001CE820`) and sent by an INTC handler, not by a DMA chain:
- `AddIntcHandler(5, 0x001CFAF8)`, cause 5 = VIF1's interrupt. The VIF1
  display lists (double-buffered at 0x004E9000 / 0x00410000) carry 26
  `0x93000000` - FLUSHA with the interrupt bit - each after an MSCAL/MSCNT,
  in DMA tags' upper halves.
- At each, VIF1 stalls; `0x001CFAF8` takes the next upload of the list at
  `*(0x005C2264)` (+0x200 count, +0x204 index, +0x208 entries), clears GS
  CSR.FINISH, sends it (`0x001CEB08`), waits for FINISH, and writes
  VIF1_FBRST = 8 (STC): VIF1 goes on, the texture in GS memory before the
  draws that use it.
- VIF1_ERR is 2 (MII clear: the interrupt is on); FBRST = 1 (reset) once, in
  XAVStr's initialisation.

The host ignored bit 31 of VIFcodes; it now stalls VIF1 there (engine
commits da84382, c6838bb). The report's "VIF1: N interrupt stalls" counts
them: about 26 a frame. The first screen is the language select (English,
Français, Italiano, Deutsch, Español), from frame ~200.

Also found on the way: EE thread 2 (entry `0x0025D3E0`, created by
`0x0025D4B8`) is libkernel's helper for the i-calls (WakeupThread,
RotateThreadReadyQueue, SuspendThread from interrupts): asleep on its
semaphore by design. `*0x0026F4F4` -> `0x01FE6FE8`, a 64-bit value
(0x01139AB2 at frame 2006) the main loop `0x00119600` compares with 0x90 and
0x8732B. The pad is opened (port 0 slot 0, area `0x005CBA40`).
