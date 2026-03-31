# Space Invaders RL

Repozitar je ted rozdeleny tak, aby se dal pohodlne pouzivat na macOS i na Windows 11 a pritom zustal kompatibilni se stavajicimi modely v `models/`.

## Struktura

- `space_invaders_rl/`: hlavni Python balicek s herni logikou, Gymnasium envem, treninkem, evaluaci a benchmarkem
- `scripts/macos/`: bash skripty pro macOS
- `scripts/windows/`: PowerShell skripty pro Windows 11
- `models/`: hlavni modely, replay buffery a checkpointy
- `runs/`: logy a TensorBoard vystupy
- `docs/`: dokumentace a plan

Root repozitare je ted zamerne cistsi. Hlavni logika je pouze v `space_invaders_rl/` a spousteni je pres skripty v `scripts/` nebo pres `python -m ...`.

## Prvni spusteni po klonu

Po prvnim naklonovani repozitare staci spustit init script pro tvoji platformu. Ten:

- zkontroluje Python a zakladni predpoklady
- vytvori `.venv`, pokud jeste neexistuje
- nainstaluje zavislosti z `requirements.txt`
- pripravi potrebne slozky jako `.mplconfig`, `runs/` a `models/checkpoints`

macOS:

```bash
./scripts/macos/init.sh
```

Windows 11 PowerShell:

```powershell
.\scripts\windows\Init.ps1
```

Windows 11 CMD:

```cmd
.\scripts\windows\Init.cmd
```

Kdyz potrebujes `.venv` vytvorit znovu:

```bash
./scripts/macos/init.sh --recreate-venv
```

```powershell
.\scripts\windows\Init.ps1 --recreate-venv
```

## Zachovani modelu

Hlavni model a replay buffer zustavaji na puvodnich cestach:

- `models/space_invaders_dqn.zip`
- `models/space_invaders_dqn_replay_buffer.pkl`

Pred refaktorem byla vytvorena zaloha:

- `models/backups/`

## macOS

Spousteni:

```bash
./scripts/macos/train_long.sh
./scripts/macos/train_cycle.sh
./scripts/macos/silent_test.sh
./scripts/macos/visual_game.sh
./scripts/macos/benchmark.sh
```

Dlouhy trening je nastaveny pro vice-denni beh:

- `TIMESTEPS=100000000`
- `N_ENVS=10`
- `TRAIN_DEVICE=cpu`

Priklad s resetem replay bufferu, ale bez zahazovani modelu:

```bash
RESET_REPLAY_BUFFER=1 ./scripts/macos/train_long.sh
```

## Windows 11

Spousteni v PowerShellu z rootu repozitare:

```powershell
.\scripts\windows\Train-Long.ps1
.\scripts\windows\Train-Cycle.ps1
.\scripts\windows\Silent-Test.ps1
.\scripts\windows\Visual-Game.ps1
.\scripts\windows\Benchmark.ps1
```

Priklad s resetem replay bufferu:

```powershell
$env:RESET_REPLAY_BUFFER = "1"
.\scripts\windows\Train-Long.ps1
```

## Python entrypointy

Vsechny hlavni akce jdou spoustet i pres moduly:

```bash
python -m space_invaders_rl.game
python -m space_invaders_rl.train
python -m space_invaders_rl.evaluate
python -m space_invaders_rl.benchmark
```
