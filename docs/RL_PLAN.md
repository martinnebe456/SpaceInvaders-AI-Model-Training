# RL plán pro Space Invaders

Tento dokument slouží jako průběžná pracovní poznámka k tomu, jak z aktuální hry udělat prostředí pro reinforcement learning a natrénovat neuronovou síť, která se naučí hrát.

## Aktuální stav repozitáře

Repozitář je zatím velmi malý:

- `space_invaders.py`
- `__pycache__/...`

Celá hra je tedy soustředěná v jednom souboru `space_invaders.py`.

## Co už je pro RL připravené

Aktuální implementace už má překvapivě dobrý základ pro trénování:

- hra umí běžet v `human_mode=True` i `human_mode=False`
- existuje `reset()`, které vrací stav hry
- existuje `step(action)`, které vrací stav, reward, done a info
- existuje `get_state_vector()`, které převádí hru na numerický vstup pro neuronovou síť

To znamená, že nemusíme začínat od nuly. Spíš budeme stávající hru upravovat tak, aby byla korektním RL prostředím.

## Důležité problémy, které je potřeba opravit před tréninkem

Než se začne s Gymnasium wrapperem a trénováním modelu, je potřeba vyřešit několik věcí v herní logice.

### 1. Cooldown střelby je závislý na reálném čase

V AI režimu se střelba řídí přes `pygame.time.get_ticks()`. To je problém, protože reinforcement learning má být řízen po krocích nebo framech, ne podle wall-clock času.

Co udělat:

- převést cooldown střelby z milisekund na počet frameů
- ideálně zavést `frame_count` nebo `step_count`
- střelbu kontrolovat přes rozdíl kroků, ne přes čas systému

### 2. `death_timer` se v AI režimu nesnižuje

Po zásahu hráče se nastaví `death_timer`, ale v `step()` se jeho hodnota nesnižuje. Snižování dnes existuje jen v lidské herní smyčce.

Důsledek:

- po prvním zásahu může být agent v AI režimu prakticky trvale nesmrtelný

Co udělat:

- přidat decrement `death_timer` i do logiky používané v `step()`

### 3. Obtížnost se ukládá do globální proměnné

`ENEMY_SHOOT_CHANCE` se při dalším levelu mění globálně. To je nepříjemné pro reprodukovatelnost i pro `reset()`.

Důsledek:

- jedna epizoda může ovlivnit další epizodu

Co udělat:

- přesunout obtížnost do instance hry
- při `reset()` vracet všechny hodnoty do výchozího stavu

### 4. Pozorování není úplně úplné

Současný state vector je použitelný, ale pro ideální RL stav v něm chybí některé důležité informace:

- cooldown střelby
- `death_timer`
- případně počet kroků v epizodě

Navíc se ukládají jen první 4 nepřátelské střely, i když hra jich může mít víc.

Co udělat:

- rozšířit stavový vektor o chybějící interní proměnné
- rozhodnout, zda chceme bullet list omezit, nebo agregovat jiným způsobem

### 5. Není jasně definované vítězství epizody

Proměnná `victory` se používá v reward logice, ale hra po vyčištění levelu rovnou přejde na další level.

Co udělat:

- vědomě rozhodnout, co je konec epizody
- pro první verzi doporučeno: `1 episode = 1 level`

## Doporučená architektura

Nedoporučuje se přepsat celou hru rovnou do Gymnasium API. Lepší je rozdělit věc na dvě vrstvy:

### Vrstva 1: herní logika

Nechat `SpaceInvadersGame` jako interní engine hry.

Zodpovědnost:

- pravidla hry
- aktualizace objektů
- kolize
- výpočet stavu
- vykreslení

### Vrstva 2: Gymnasium wrapper

Vytvořit nový soubor, například:

- `space_invaders_env.py`

Tento wrapper bude dělat jen tohle:

- dědit z `gymnasium.Env`
- držet instanci `SpaceInvadersGame`
- mapovat `reset()` a `step()` do Gymnasium API
- definovat `action_space`
- definovat `observation_space`
- vracet `terminated` a `truncated`

## Doporučený postup implementace

### Fáze 1: stabilizace herní logiky pro RL

Upravit `space_invaders.py` tak, aby:

- hra byla plně step-based
- šla seedovat
- byla determinističtější
- nevyužívala reálný čas pro mechaniky důležité pro AI

Konkrétně:

- přidat `seed` podporu
- použít vlastní RNG, například `random.Random(seed)`
- převést cooldown na framy
- opravovat `death_timer` v AI cestě
- oddělit globální obtížnost od globálních konstant

### Fáze 2: vytvoření Gymnasium environmentu

V novém souboru vytvořit něco ve stylu:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from space_invaders import SpaceInvadersGame


class SpaceInvadersEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, max_steps=5000):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.game = SpaceInvadersGame(human_mode=False)
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(190,),
            dtype=np.float32,
        )
```

Poznámka:

- finální rozměr `observation_space` se může změnit, pokud rozšíříme state vector

### Fáze 3: definice epizody

Pro první verzi doporučeno:

- `terminated=True`, když hráč umře
- `terminated=True`, když agent vyčistí celý level
- `truncated=True`, když epizoda přesáhne `max_steps`

To je jednodušší než nekonečné levelování v jedné epizodě.

### Fáze 4: kontrola environmentu

Ještě před tréninkem spouštět kontrolu:

```python
from stable_baselines3.common.env_checker import check_env

env = SpaceInvadersEnv()
check_env(env)
```

To odhalí spoustu chyb dřív, než člověk stráví hodiny tréninkem.

### Fáze 5: první trénink

Na začátek doporučeno:

- používat vektorový stav
- nepoužívat obraz jako vstup
- začít s `DQN`

Proč:

- akční prostor je diskrétní
- pozorování je číselný vektor
- debug je mnohem jednodušší než u CNN nad pixely

## Jaký algoritmus zvolit jako první

Pro první funkční verzi:

- `Stable-Baselines3`
- `DQN`
- politika `MlpPolicy`

To je nejjednodušší a pro tuto úlohu velmi rozumný začátek.

Později lze zkusit:

- `PPO`
- obrazový vstup místo vektoru
- vlastní síť
- vlastní trénovací skript bez SB3

Ale jako první krok je lepší minimalizovat počet pohyblivých částí.

## Doporučení k reward funkci

První verzi rewardu nepřekombinovat.

Doporučený start:

- reward za získané score
- penalizace za smrt nebo ztrátu života
- bonus za vyčištění levelu
- malá step penalty, aby agent dlouho nestál

Pracovní návrh:

```text
reward = score_delta / 10
+ bonus za clear level
- penalizace za lose
- malá step penalty
```

Pozor:

- příliš agresivní reward shaping může vést k tomu, že se agent naučí něco jiného, než chceme

## Doporučené soubory, které pravděpodobně vzniknou

Pravděpodobná struktura další práce:

- `space_invaders.py`
- `space_invaders_env.py`
- `train_dqn.py`
- `evaluate_agent.py`
- `requirements.txt`
- případně `models/` pro uložené checkpointy

## Kde hledat návody a inspiraci

K 30. 3. 2026 je lepší používat `Gymnasium`, ne původní OpenAI Gym.

### Oficiální a velmi užitečné zdroje

- Gymnasium custom environment tutorial:
  https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
- Gymnasium create custom env:
  https://gymnasium.farama.org/main/introduction/create_custom_env/
- Gymnasium migration guide:
  https://gymnasium.farama.org/main/introduction/migration_guide/
- OpenAI Gym README s upozorněním na přesun do Gymnasium:
  https://github.com/openai/gym
- Stable-Baselines3 custom env guide:
  https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
- Stable-Baselines3 DQN docs:
  https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
- Stable-Baselines3 RL tips:
  https://stable-baselines3.readthedocs.io/en/v2.2.1/guide/rl_tips.html
- CleanRL:
  https://docs.cleanrl.dev/

### Doporučené vyhledávací dotazy

Používat hlavně tyto dotazy:

- `gymnasium custom environment pygame`
- `stable-baselines3 custom env check_env`
- `stable-baselines3 dqn custom environment`
- `cleanrl dqn single file`

## Praktický plán krok za krokem

### Nejbližší doporučený postup

1. Opravit herní logiku, aby byla vhodná pro RL.
2. Přidat seedování a frame-based cooldown.
3. Dopsat Gymnasium wrapper.
4. Spustit `check_env`.
5. Udělat jednoduchý `train_dqn.py`.
6. Otestovat, že agent aspoň občas získá body.
7. Teprve potom ladit reward a architekturu sítě.

## Co je teď nejlepší další krok

Nejrozumnější další krok je:

- nejdřív upravit `space_invaders.py` pro korektní RL chování
- až potom vytvořit `space_invaders_env.py`

Pokud bychom to udělali opačně, jen bychom zabalili současné problémy do Gym wrapperu a debugging by byl zbytečně těžký.

## Poznámka pro nás oba

Tento dokument je záměrně psaný jako pracovní plán, ne jako finální dokumentace. Klidně do něj budeme průběžně dopisovat:

- co už je hotové
- co se ukázalo jako slepá cesta
- jaké hyperparametry fungovaly
- jaké reward shaping varianty se osvědčily

