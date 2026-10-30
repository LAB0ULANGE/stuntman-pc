Stuntman - PC port, version {version}
==================================================

An unofficial, fan-made native Windows port of Stuntman (PS2). The game's
PS2 program was translated to C++ and compiled for the PC; it runs on your
computer directly, not in an emulator.

This download contains NO game content. You need your own copy of the
game: an ISO disc image you made from your own disc - the European PS2
release, SLES-50288 (executable SLES_502.88).
Nothing of the game - no textures, sounds, music, movies or data - is
included, and none is downloaded: everything comes from your disc image.

Not affiliated with, endorsed by or sponsored by the game's publisher or
developer. Its trademarks belong to their owners.

This is an ALPHA: expect bugs.


What you need
-------------

* Windows 10 or 11, 64-bit
* A processor with AVX2
* A graphics card with a Vulkan 1.3 driver
* Your own disc image of Stuntman (SLES_502.88)

The game runs at 25 pictures a second, as on the console. To hold that:

                  Minimum (resolution=original)   Recommended (resolution=3x)
  Processor       4 cores, fast per core:         Intel Core i7-10700K,
                  Intel Core i5-12400,            AMD Ryzen 5 5600X or better
                  AMD Ryzen 5 5600 or better
  Graphics card   NVIDIA RTX 3060 /               NVIDIA RTX 3070 /
                  AMD RX 6600 (estimate)          AMD RX 6700 XT
  Memory          8 GB                            8 GB

Tested on an Intel Core i7-10700K with an NVIDIA RTX 3070: 25 pictures a
second at every resolution, from 4 cores up (2 cores: about 15). The
heaviest part runs on one core, so speed per core matters more than the
number of cores. The minimum graphics card is an estimate: reports welcome.


Getting started
---------------

1. Unzip this folder somewhere you can write to (not Program Files): the
   game keeps its saves and settings beside the executable.
2. Run the executable. The first time, it asks for your disc image and
   checks that it is the right disc.

Alt+Enter switches between a window and fullscreen.


Settings
--------

pc_settings.ini, beside the executable (open it with Notepad; the game
reads it when it starts):

  resolution=2x       the 3D drawn at the PS2's size times this: original,
                      2x, 3x... (higher is sharper, and heavier for the
                      graphics card)
  fullscreen=on       start in fullscreen (Alt+Enter switches too)
  widescreen=on       a wider view of the scene, at the shape of aspect=
  aspect=16:9         16:9, 21:9 or 32:9 (ultrawide), or auto (the
                      window's or the screen's own shape); take both
                      lines out for the original 4:3
  antialiasing=fxaa   smoother edges (off to take it out)
  sharpen=25          0 (none) to 100 (sharpest)
  shadows=off         no shadows (everything lit, as before they were fixed)

The other lines in it are the port's speed settings: leave them as they
are.


Controls
--------

A game controller works as a DualShock 2 (with rumble); its right trigger
accelerates and its left one brakes (right_trigger= and left_trigger= in
pc_settings.ini: remove them for L2 and R2). Keyboard: arrow keys
the D-pad, X Cross, C Circle, Z Square, V Triangle, Enter Start, Backspace
Select, Q / E L1 / R1, 1 / 3 L2 / R2.


Files beside the executable
---------------------------

  pc_settings.ini   your settings, and where your disc image is
  memcard\          your memory card: the game's saves
  *.log             what the last run did (useful in a bug report)


Known issues
------------

* With widescreen on, the 2D (gauges, timer, menus) is stretched sideways:
  it is drawn straight to the screen, not by the game's camera.
* The menus' animations are a little less smooth than on the console.
* Below the minimum configuration, the stunts get jerky or slow.
* Only the European disc (SLES-50288) is supported.

To report a problem: say what you did and what happened, and join
stuntman.log (beside the executable, written at every start).


Licences
--------

Third-party licences are in THIRD-PARTY-NOTICES.txt.
