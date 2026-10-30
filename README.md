# Stuntman PC

An unofficial, fan-made **native Windows port of Stuntman** (PlayStation 2,
2002). The game's PS2 program was translated to C++ and compiled for the PC:
it runs directly on your computer, not in an emulator - with higher
resolutions, widescreen and ultrawide, smoothing and your controller.

> **No game content here.** This repository and its downloads contain none of
> the game's textures, sounds, music, movies or data. You need **your own
> disc** of Stuntman: the port reads everything from an ISO image you made
> from it. Not affiliated with, endorsed by or sponsored by the game's
> publisher or developer; Stuntman is a trademark of its owner.

**Status: alpha.** Playable (the career and its stunts), with bugs left.

## Playing

1. Download `Stuntman-PC-<version>-windows-x64.zip` from the
   [Releases](../../releases) and unzip it somewhere you can write to (not
   Program Files).
2. Run `stuntman.exe`. The first time, it asks for your disc image and checks
   that it is the right disc: the **European release, SLES-50288**.
3. Play. Alt+Enter switches between a window and fullscreen. The settings are
   in `pc_settings.ini` beside the executable (see its README.txt).

### Requirements

The game runs at 25 pictures a second, as on the console. To hold that:

|               | Minimum (`resolution=original`)                         | Recommended (`resolution=3x`)                  |
|---------------|---------------------------------------------------------|------------------------------------------------|
| System        | Windows 10/11, 64-bit                                   | Windows 10/11, 64-bit                          |
| Processor     | 4 cores, fast per core: Intel Core i5-12400, AMD Ryzen 5 5600 or better | Intel Core i7-10700K, AMD Ryzen 5 5600X or better |
| Graphics card | Vulkan 1.3: NVIDIA RTX 3060 / AMD RX 6600 (estimate)    | NVIDIA RTX 3070 / AMD RX 6700 XT               |
| Memory        | 8 GB                                                    | 8 GB                                           |

Tested on an Intel Core i7-10700K with an NVIDIA RTX 3070: 25 pictures a
second at every resolution from 4 cores up (2 cores: about 15). The heaviest
part runs on one core, so speed per core matters more than the number of
cores. The minimum graphics card is an estimate - reports are welcome.

### Reporting a problem

Open an issue saying what you did and what happened, with `stuntman.log`
(beside the executable, written at every start).

## Building it yourself

The port is regenerated from your own disc by the tools: Windows, Visual
Studio 2022 with C++, Python 3.10+ and your disc image. See `LISEZMOI.txt`
(the steps) and `engine/README.md` (how the engine works). The game's
translated code (`games/stuntman/src/`) is never in this repository.

## Credits

- The engine, [ps2-port-engine](https://github.com/danielgomesvieira2000/ps2-port-engine)
  by Daniel Gomes Vieira (MIT), extended here for Stuntman.
- Built with [Claude Code](https://claude.com/claude-code).
- Libraries built in: SDL 3, volk, Vulkan-Headers, glslang
  (licences in the release's THIRD-PARTY-NOTICES.txt).

## Licence

The port's own code is under the MIT licence (`LICENSE`; the engine's in
`engine/LICENSE`). It does not cover the game, which belongs to its owners.

---

**En français :** portage PC non officiel de Stuntman (PS2). Aucun contenu du
jeu n'est fourni : il faut votre propre disque (version européenne
SLES-50288) en image ISO. Téléchargez le zip dans les Releases, décompressez,
lancez `stuntman.exe` et choisissez votre image disque au premier lancement.
