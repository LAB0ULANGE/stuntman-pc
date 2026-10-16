# Scripted runs

Each `.args` file is a set of port options that drives the game to a known
point. With `--virtual-time` the run is the same on any machine, under any
load: the same presses land on the same frames and the pictures come out
byte-identical - a regression test, and a way to get somewhere in the game
without playing.

```sh
<exe> <executable> --disc <iso> $(cat games/stuntman/port/runs/<run>.args) --frames out/ --frames-every 200
python engine/tools/regress.py --game games/stuntman <run> --gpu            # check against the recorded CRCs
python engine/tools/regress.py --game games/stuntman <run> --gpu --update   # record them (an intended change only)
```

Record runs that reach the game's main places - boot, the menus, gameplay -
and say here what each reaches and at which frames:

| file | reaches | frames |
| --- | --- | --- |
| `language_fr.args` | the language select (from ~150); meant to choose FRANCAIS (down@280, cross@300) - **not yet**: the presses reach the game's input (0x01F30610 +0x20/+0x24) but the menu does not act on them; no CRCs recorded | 0-3000 |

`python engine/tools/regress_all.py` runs every one of them (and every other
game's, given their repositories) - before an engine version is tagged.
