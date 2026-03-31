"""
Space Invaders – plná kopie klasické arkádové hry.

Struktura je navržena tak, aby stav hry bylo snadné exportovat
jako vektor čísel pro vstupní vrstvu neuronové sítě (viz metoda
SpaceInvadersGame.get_state_vector()).
"""

import math
import random
import pygame

# ---------------------------------------------------------------------------
# Konstanty
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Barvy
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
ORANGE = (255, 165, 0)

# Hráč
PLAYER_WIDTH = 48
PLAYER_HEIGHT = 32
PLAYER_SPEED = 5
PLAYER_Y = SCREEN_HEIGHT - 60

# Střely hráče
BULLET_WIDTH = 4
BULLET_HEIGHT = 12
BULLET_SPEED = 8
BULLET_COOLDOWN_MS = 300  # ms
BULLET_COOLDOWN_FRAMES = max(1, round(BULLET_COOLDOWN_MS * FPS / 1000))
PLAYER_INVULNERABILITY_FRAMES = 90

# Nepřátelé
ENEMY_ROWS = 5
ENEMY_COLS = 11
ENEMY_WIDTH = 32
ENEMY_HEIGHT = 24
ENEMY_PADDING_X = 16
ENEMY_PADDING_Y = 12
ENEMY_START_X = 60
ENEMY_START_Y = 80
ENEMY_BASE_SPEED = 1.0
ENEMY_SPEED_INCREASE = 0.15  # rychlost roste s každým zabitým
ENEMY_DROP = 16  # posun dolů při odrazu od stěny

# Střely nepřátel
ENEMY_BULLET_WIDTH = 4
ENEMY_BULLET_HEIGHT = 12
ENEMY_BULLET_SPEED = 4
ENEMY_SHOOT_CHANCE = 0.002  # šance na výstřel za frame za sloupec

# Mystery ship
MYSTERY_WIDTH = 48
MYSTERY_HEIGHT = 20
MYSTERY_SPEED = 3
MYSTERY_SCORES = [50, 100, 150, 300]
MYSTERY_INTERVAL = 600  # framů mezi mystery ship

# Bunkry (shields)
BUNKER_COUNT = 4
BUNKER_BLOCK_SIZE = 4
BUNKER_Y = SCREEN_HEIGHT - 140
MAX_OBS_ENEMY_BULLETS = 8
MAX_OBS_PLAYER_BULLETS = 3
MAX_TRACKED_FRAMES = 5000

# Reward shaping pro RL
STEP_PENALTY = 0.002
LIFE_LOSS_PENALTY = 5.0
GAME_OVER_PENALTY = 12.0
VICTORY_BONUS = 35.0
PLAYER_MISSED_SHOT_PENALTY = 0.03
PLAYER_TARGETED_SHOT_BONUS = 0.04
PLAYER_BLOCKED_SHOT_PENALTY = 0.10
PLAYER_BUNKER_HIT_PENALTY = 0.20
MOVE_TOWARD_CLEAR_SHOT_BONUS = 0.015
MOVE_AWAY_FROM_CLEAR_SHOT_PENALTY = 0.005
IDLE_WITH_CLEAR_SHOT_PENALTY = 0.01

# Skóre za řady (od spodní řady nahoru)
ROW_SCORES = [10, 10, 20, 20, 30]

# ---------------------------------------------------------------------------
# Pomocné funkce pro vykreslování pixelartových tvarů
# ---------------------------------------------------------------------------

def draw_player(surface, x, y, color=GREEN):
    """Nakreslí hráčovu loď (klasický tvar kanónu)."""
    # Tělo
    pygame.draw.rect(surface, color, (x + 4, y + 12, 40, 20))
    # Hlaveň
    pygame.draw.rect(surface, color, (x + 20, y, 8, 16))
    # Podstavec
    pygame.draw.rect(surface, color, (x, y + 24, 48, 8))


def draw_enemy_type_a(surface, x, y, color, frame):
    """Horní řada – malý nepřítel (squid)."""
    w, h = 24, 18
    ox = x + (ENEMY_WIDTH - w) // 2
    oy = y + (ENEMY_HEIGHT - h) // 2
    # Tělo
    pygame.draw.rect(surface, color, (ox + 8, oy, 8, 6))
    pygame.draw.rect(surface, color, (ox + 4, oy + 6, 16, 6))
    pygame.draw.rect(surface, color, (ox, oy + 12, 24, 6))
    # Oči
    pygame.draw.rect(surface, BLACK, (ox + 8, oy + 12, 4, 4))
    pygame.draw.rect(surface, BLACK, (ox + 14, oy + 12, 4, 4))
    # Nožky
    if frame == 0:
        pygame.draw.rect(surface, color, (ox + 2, oy + 18, 4, 4))
        pygame.draw.rect(surface, color, (ox + 18, oy + 18, 4, 4))
    else:
        pygame.draw.rect(surface, color, (ox - 2, oy + 16, 4, 4))
        pygame.draw.rect(surface, color, (ox + 22, oy + 16, 4, 4))


def draw_enemy_type_b(surface, x, y, color, frame):
    """Střední řady – střední nepřítel (crab)."""
    w, h = 28, 20
    ox = x + (ENEMY_WIDTH - w) // 2
    oy = y + (ENEMY_HEIGHT - h) // 2
    # Tělo
    pygame.draw.rect(surface, color, (ox + 4, oy, 20, 8))
    pygame.draw.rect(surface, color, (ox, oy + 8, 28, 8))
    # Oči
    pygame.draw.rect(surface, BLACK, (ox + 6, oy + 8, 4, 4))
    pygame.draw.rect(surface, BLACK, (ox + 18, oy + 8, 4, 4))
    # Spodek
    pygame.draw.rect(surface, color, (ox + 4, oy + 16, 8, 4))
    pygame.draw.rect(surface, color, (ox + 16, oy + 16, 8, 4))
    if frame == 0:
        pygame.draw.rect(surface, color, (ox, oy + 14, 4, 4))
        pygame.draw.rect(surface, color, (ox + 24, oy + 14, 4, 4))
    else:
        pygame.draw.rect(surface, color, (ox - 2, oy + 16, 4, 4))
        pygame.draw.rect(surface, color, (ox + 26, oy + 16, 4, 4))


def draw_enemy_type_c(surface, x, y, color, frame):
    """Spodní řady – velký nepřítel (octopus)."""
    w, h = 24, 18
    ox = x + (ENEMY_WIDTH - w) // 2
    oy = y + (ENEMY_HEIGHT - h) // 2
    pygame.draw.rect(surface, color, (ox + 4, oy, 16, 6))
    pygame.draw.rect(surface, color, (ox, oy + 6, 24, 8))
    pygame.draw.rect(surface, BLACK, (ox + 4, oy + 8, 4, 4))
    pygame.draw.rect(surface, BLACK, (ox + 16, oy + 8, 4, 4))
    pygame.draw.rect(surface, color, (ox + 2, oy + 14, 6, 4))
    pygame.draw.rect(surface, color, (ox + 16, oy + 14, 6, 4))
    if frame == 0:
        pygame.draw.rect(surface, color, (ox + 10, oy + 14, 4, 4))
    else:
        pygame.draw.rect(surface, color, (ox + 8, oy + 16, 8, 4))


ENEMY_DRAW_FUNCS = [draw_enemy_type_c, draw_enemy_type_c,
                    draw_enemy_type_b, draw_enemy_type_b,
                    draw_enemy_type_a]

ENEMY_COLORS = [WHITE, WHITE, CYAN, CYAN, MAGENTA]


def create_bunker_shape():
    """Vrátí 2D pole (mřížku) tvaru bunkru."""
    # Klasický tvar bunkru – oblouk nahoře, výřez dole uprostřed
    cols = 22
    rows = 16
    shape = []
    for r in range(rows):
        row = []
        for c in range(cols):
            # Horní oblouk
            cx = cols / 2
            cy = 4
            dist = math.sqrt((c - cx) ** 2 + (r - cy) ** 2)
            if r < 4 and dist > 10:
                row.append(0)
            elif r >= 12 and 8 <= c <= 13:
                # Výřez dole
                row.append(0)
            else:
                row.append(1)
        shape.append(row)
    return shape


BUNKER_SHAPE = create_bunker_shape()
BUNKER_MAX_BLOCKS = sum(sum(row) for row in BUNKER_SHAPE)
STATE_VECTOR_SIZE = (
    7
    + ENEMY_ROWS * ENEMY_COLS * 3
    + 1
    + 2
    + MAX_OBS_ENEMY_BULLETS * 2
    + MAX_OBS_PLAYER_BULLETS * 2
    + BUNKER_COUNT
)


# ---------------------------------------------------------------------------
# Herní třídy
# ---------------------------------------------------------------------------

class Player:
    def __init__(self):
        self.x = SCREEN_WIDTH // 2 - PLAYER_WIDTH // 2
        self.y = PLAYER_Y
        self.width = PLAYER_WIDTH
        self.height = PLAYER_HEIGHT
        self.speed = PLAYER_SPEED
        self.last_shot_frame = -BULLET_COOLDOWN_FRAMES

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def move(self, dx):
        self.x = max(0, min(SCREEN_WIDTH - self.width, self.x + dx * self.speed))

    def draw(self, surface):
        draw_player(surface, self.x, self.y)


class Bullet:
    def __init__(self, x, y, dy, width=BULLET_WIDTH, height=BULLET_HEIGHT,
                 color=WHITE, owner="neutral"):
        self.x = x
        self.y = y
        self.dy = dy
        self.width = width
        self.height = height
        self.color = color
        self.owner = owner
        self.active = True
        self.hit_target = False
        self.predicted_bunker_hit = False

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self):
        self.y += self.dy
        if self.y < -self.height or self.y > SCREEN_HEIGHT:
            self.active = False

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)


class Enemy:
    def __init__(self, row, col, x, y):
        self.row = row
        self.col = col
        self.x = x
        self.y = y
        self.alive = True
        self.score = ROW_SCORES[row]

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, ENEMY_WIDTH, ENEMY_HEIGHT)

    def draw(self, surface, frame):
        if self.alive:
            ENEMY_DRAW_FUNCS[self.row](surface, self.x, self.y,
                                       ENEMY_COLORS[self.row], frame)


class EnemyFormation:
    def __init__(self):
        self.enemies = []
        self.direction = 1  # 1 = doprava, -1 = doleva
        self.speed = ENEMY_BASE_SPEED
        self.move_timer = 0
        self.move_interval = 40  # framů mezi pohyby
        self.anim_frame = 0
        self._create_formation()

    def _create_formation(self):
        self.enemies = []
        for row in range(ENEMY_ROWS):
            for col in range(ENEMY_COLS):
                x = ENEMY_START_X + col * (ENEMY_WIDTH + ENEMY_PADDING_X)
                y = ENEMY_START_Y + row * (ENEMY_HEIGHT + ENEMY_PADDING_Y)
                self.enemies.append(Enemy(row, col, x, y))

    def alive_enemies(self):
        return [e for e in self.enemies if e.alive]

    def alive_count(self):
        return len(self.alive_enemies())

    def update(self):
        self.move_timer += 1
        alive = self.alive_enemies()
        if not alive:
            return

        # Dynamická rychlost – čím méně nepřátel, tím rychleji
        total = ENEMY_ROWS * ENEMY_COLS
        killed = total - len(alive)
        self.move_interval = max(2, 40 - killed)

        if self.move_timer >= self.move_interval:
            self.move_timer = 0
            self.anim_frame = 1 - self.anim_frame

            # Zkontroluj, jestli formace narazí na stěnu
            should_drop = False
            for e in alive:
                next_x = e.x + self.direction * self.speed
                if next_x <= 0 or next_x + ENEMY_WIDTH >= SCREEN_WIDTH:
                    should_drop = True
                    break

            if should_drop:
                for e in alive:
                    e.y += ENEMY_DROP
                self.direction *= -1
            else:
                for e in alive:
                    e.x += self.direction * self.speed

    def get_bottom_enemies(self):
        """Vrátí nejnižšího živého nepřítele v každém sloupci."""
        columns = {}
        for e in self.enemies:
            if e.alive:
                if e.col not in columns or e.row > columns[e.col].row:
                    columns[e.col] = e
        return list(columns.values())

    def lowest_y(self):
        alive = self.alive_enemies()
        if not alive:
            return 0
        return max(e.y + ENEMY_HEIGHT for e in alive)

    def draw(self, surface):
        for e in self.enemies:
            e.draw(surface, self.anim_frame)


class MysteryShip:
    def __init__(self, rng):
        self.rng = rng
        self.active = False
        self.x = 0
        self.y = 40
        self.width = MYSTERY_WIDTH
        self.height = MYSTERY_HEIGHT
        self.direction = 1
        self.score = 100
        self.timer = MYSTERY_INTERVAL

    def spawn(self):
        self.active = True
        self.score = self.rng.choice(MYSTERY_SCORES)
        if self.rng.random() < 0.5:
            self.x = -MYSTERY_WIDTH
            self.direction = 1
        else:
            self.x = SCREEN_WIDTH
            self.direction = -1

    def update(self):
        if self.active:
            self.x += MYSTERY_SPEED * self.direction
            if self.x < -MYSTERY_WIDTH or self.x > SCREEN_WIDTH:
                self.active = False
        else:
            self.timer -= 1
            if self.timer <= 0:
                self.spawn()
                self.timer = MYSTERY_INTERVAL + self.rng.randint(-100, 200)

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, surface):
        if self.active:
            pygame.draw.rect(surface, RED, (self.x + 8, self.y, 32, 8))
            pygame.draw.rect(surface, RED, (self.x, self.y + 8, 48, 8))
            pygame.draw.rect(surface, RED, (self.x + 4, self.y + 16, 8, 4))
            pygame.draw.rect(surface, RED, (self.x + 36, self.y + 16, 8, 4))
            pygame.draw.rect(surface, RED, (self.x + 18, self.y - 4, 12, 6))


class Bunker:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.blocks = []
        for r, row in enumerate(BUNKER_SHAPE):
            for c, val in enumerate(row):
                if val:
                    bx = x + c * BUNKER_BLOCK_SIZE
                    by = y + r * BUNKER_BLOCK_SIZE
                    self.blocks.append(pygame.Rect(bx, by, BUNKER_BLOCK_SIZE,
                                                   BUNKER_BLOCK_SIZE))

    def check_collision(self, rect):
        hit = False
        new_blocks = []
        for b in self.blocks:
            if b.colliderect(rect):
                hit = True
            else:
                new_blocks.append(b)
        self.blocks = new_blocks
        return hit

    def damage_area(self, cx, cy, radius=8):
        """Odstraní bloky v okruhu kolem bodu – pro explozi."""
        self.blocks = [b for b in self.blocks
                       if math.sqrt((b.centerx - cx) ** 2 +
                                    (b.centery - cy) ** 2) > radius]

    def draw(self, surface):
        for b in self.blocks:
            pygame.draw.rect(surface, GREEN, b)


# ---------------------------------------------------------------------------
# Hlavní herní třída
# ---------------------------------------------------------------------------

class SpaceInvadersGame:
    """
    Hlavní třída hry. Může být ovládána klávesnicí (human mode)
    nebo programově přes metodu step() pro AI/neuronovou síť.
    """

    def __init__(self, human_mode=True, seed=None, single_level_mode=False):
        self.human_mode = human_mode
        self.single_level_mode = single_level_mode
        self.rng = random.Random()
        self.random_seed = None

        if human_mode:
            if not pygame.get_init():
                pygame.init()
        else:
            if not pygame.font.get_init():
                pygame.font.init()

        if human_mode:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Space Invaders")
        else:
            # Headless mode pro trénink AI
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20)
        self.big_font = pygame.font.SysFont("monospace", 48, bold=True)
        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.random_seed = seed
        self.rng.seed(seed)
        return seed

    def reset(self, seed=None):
        """Resetuje hru na počáteční stav. Vrátí state vector."""
        if seed is not None:
            self.seed(seed)
        self.player = Player()
        self.formation = EnemyFormation()
        self.mystery = MysteryShip(self.rng)
        self.player_bullets = []
        self.enemy_bullets = []
        self.bunkers = self._create_bunkers()
        self.score = 0
        self.lives = 3
        self.level = 1
        self.enemy_shoot_chance = ENEMY_SHOOT_CHANCE
        self.game_over = False
        self.victory = False
        self.death_timer = 0
        self.score_popups = []  # (text, x, y, timer)
        self.frame_count = 0
        self.reward_adjustment = 0.0
        self.player_shots_fired = 0
        self.player_shots_missed = 0
        self.player_targeted_shots = 0
        self.player_blocked_shots = 0
        self.player_bunker_hits = 0
        return self.get_state_vector()

    def _create_bunkers(self):
        bunkers = []
        spacing = SCREEN_WIDTH // (BUNKER_COUNT + 1)
        for i in range(BUNKER_COUNT):
            bx = spacing * (i + 1) - 44
            bunkers.append(Bunker(bx, BUNKER_Y))
        return bunkers

    # -------------------------------------------------------------------
    # AI / Neuronová síť rozhraní
    # -------------------------------------------------------------------

    def get_state_vector(self):
        """
        Vrátí stav hry jako plochý seznam čísel (floats) pro vstup
        do neuronové sítě.

        Vektor je z většiny normalizovaný do intervalu 0..1 a obsahuje:
        - informace o hráči včetně cooldownu střelby
        - stav celé formace nepřátel
        - mystery ship
        - fixní počet slotů pro střely nepřátel a hráče
        - stav bunkrů

        Celková velikost vektoru je STATE_VECTOR_SIZE.
        """
        state = []
        cooldown_remaining = max(
            0,
            BULLET_COOLDOWN_FRAMES
            - (self.frame_count - self.player.last_shot_frame),
        )

        # Hráč
        state.append(self.player.x / SCREEN_WIDTH)
        state.append(self.lives / 3.0)
        state.append(min(self.score / 10000.0, 1.0))
        state.append(len(self.player_bullets) / MAX_OBS_PLAYER_BULLETS)
        state.append(cooldown_remaining / BULLET_COOLDOWN_FRAMES)
        state.append(self.death_timer / PLAYER_INVULNERABILITY_FRAMES)
        state.append(min(self.frame_count / MAX_TRACKED_FRAMES, 1.0))

        # Nepřátelé – stav, pozice
        for e in self.formation.enemies:
            state.append(1.0 if e.alive else 0.0)
        for e in self.formation.enemies:
            state.append(e.x / SCREEN_WIDTH)
        for e in self.formation.enemies:
            state.append(e.y / SCREEN_HEIGHT)

        # Formace
        state.append(1.0 if self.formation.direction > 0 else 0.0)

        # Mystery ship
        state.append(1.0 if self.mystery.active else 0.0)
        state.append(self.mystery.x / SCREEN_WIDTH if self.mystery.active
                     else 0.0)

        # Střely nepřátel (fixní počet slotů, seřazené od nejnižší střely)
        tracked_enemy_bullets = sorted(
            self.enemy_bullets,
            key=lambda b: (-b.y, abs(b.x - self.player.x)),
        )
        for i in range(MAX_OBS_ENEMY_BULLETS):
            if i < len(tracked_enemy_bullets):
                b = tracked_enemy_bullets[i]
                state.append(b.x / SCREEN_WIDTH)
                state.append(b.y / SCREEN_HEIGHT)
            else:
                state.append(0.0)
                state.append(0.0)

        # Střely hráče (fixní počet slotů, seřazené od nejvýš letící)
        tracked_player_bullets = sorted(self.player_bullets, key=lambda b: b.y)
        for i in range(MAX_OBS_PLAYER_BULLETS):
            if i < len(tracked_player_bullets):
                b = tracked_player_bullets[i]
                state.append(b.x / SCREEN_WIDTH)
                state.append(b.y / SCREEN_HEIGHT)
            else:
                state.append(0.0)
                state.append(0.0)

        # Bunkry
        for bunker in self.bunkers:
            state.append(len(bunker.blocks) / BUNKER_MAX_BLOCKS)

        return state

    def get_info(self):
        return {
            "score": self.score,
            "lives": self.lives,
            "level": self.level,
            "enemies_alive": self.formation.alive_count(),
            "victory": self.victory,
            "game_over": self.game_over,
            "frame_count": self.frame_count,
            "enemy_shoot_chance": self.enemy_shoot_chance,
            "player_shots_fired": self.player_shots_fired,
            "player_shots_missed": self.player_shots_missed,
            "player_targeted_shots": self.player_targeted_shots,
            "player_blocked_shots": self.player_blocked_shots,
            "player_bunker_hits": self.player_bunker_hits,
        }

    def step(self, action):
        """
        Provede jeden krok hry s danou akcí.

        Akce (int):
            0 = nic
            1 = doleva
            2 = doprava
            3 = střelba
            4 = doleva + střelba
            5 = doprava + střelba

        Vrátí: (state_vector, reward, done, info)
        """
        self.frame_count += 1
        self.reward_adjustment = 0.0
        old_score = self.score
        old_lives = self.lives
        old_bullet_x = self._get_player_bullet_x()
        had_clear_shot = self._has_clear_shot_from_position(old_bullet_x)
        nearest_clear_lane = self._get_nearest_clear_shot_x(old_bullet_x)

        # Převod akce na vstupy
        move = 0
        shoot = False
        if action == 1:
            move = -1
        elif action == 2:
            move = 1
        elif action == 3:
            shoot = True
        elif action == 4:
            move = -1
            shoot = True
        elif action == 5:
            move = 1
            shoot = True

        # Pohyb hráče
        if move != 0:
            self.player.move(move)
            self._reward_movement_toward_clear_shot(
                old_bullet_x,
                nearest_clear_lane,
                had_clear_shot,
            )

        if shoot:
            self._apply_idle_clear_shot_penalty(False)
        else:
            self._apply_idle_clear_shot_penalty(True)

        # Střelba
        if shoot:
            self._try_player_shoot()

        # Aktualizace herních objektů
        self._update_bullets()
        self.formation.update()
        self.mystery.update()
        self._enemy_shoot()
        self._check_collisions()
        self._tick_timers()
        self._check_win_lose()

        # Reward
        reward = (self.score - old_score) / 10.0
        reward -= STEP_PENALTY
        reward += self.reward_adjustment
        if self.lives < old_lives:
            reward -= LIFE_LOSS_PENALTY * (old_lives - self.lives)
        if self.game_over and not self.victory:
            reward -= GAME_OVER_PENALTY
        if self.victory:
            reward += VICTORY_BONUS

        done = self.game_over or self.victory
        return self.get_state_vector(), reward, done, self.get_info()

    # -------------------------------------------------------------------
    # Herní logika
    # -------------------------------------------------------------------

    def _update_bullets(self):
        for b in self.player_bullets:
            b.update()
            if not b.active and not b.hit_target:
                self.reward_adjustment -= PLAYER_MISSED_SHOT_PENALTY
                self.player_shots_missed += 1
        for b in self.enemy_bullets:
            b.update()
        self.player_bullets = [b for b in self.player_bullets if b.active]
        self.enemy_bullets = [b for b in self.enemy_bullets if b.active]

    def _enemy_shoot(self):
        bottom = self.formation.get_bottom_enemies()
        for e in bottom:
            if self.rng.random() < self.enemy_shoot_chance:
                bx = e.x + ENEMY_WIDTH // 2 - ENEMY_BULLET_WIDTH // 2
                by = e.y + ENEMY_HEIGHT
                self.enemy_bullets.append(
                    Bullet(bx, by, ENEMY_BULLET_SPEED,
                           ENEMY_BULLET_WIDTH, ENEMY_BULLET_HEIGHT,
                           color=YELLOW, owner="enemy"))

    def _check_collisions(self):
        # Střely hráče vs nepřátelé
        for b in self.player_bullets:
            if not b.active:
                continue
            for e in self.formation.enemies:
                if e.alive and b.rect.colliderect(e.rect):
                    e.alive = False
                    b.active = False
                    b.hit_target = True
                    self.score += e.score
                    self.score_popups.append(
                        (str(e.score), e.x, e.y, 30))
                    break

        # Střely hráče vs mystery ship
        if self.mystery.active:
            for b in self.player_bullets:
                if not b.active:
                    continue
                if b.rect.colliderect(self.mystery.rect):
                    b.active = False
                    b.hit_target = True
                    self.score += self.mystery.score
                    self.score_popups.append(
                        (str(self.mystery.score), self.mystery.x,
                         self.mystery.y, 45))
                    self.mystery.active = False
                    break

        # Střely nepřátel vs hráč
        if self.death_timer <= 0:
            for b in self.enemy_bullets:
                if not b.active:
                    continue
                if b.rect.colliderect(self.player.rect):
                    b.active = False
                    self.lives -= 1
                    self.death_timer = PLAYER_INVULNERABILITY_FRAMES
                    break

        # Střely vs bunkry
        for bunker in self.bunkers:
            for b in self.player_bullets + self.enemy_bullets:
                if not b.active:
                    continue
                if bunker.check_collision(b.rect):
                    bunker.damage_area(b.x + b.width // 2,
                                       b.y + b.height // 2)
                    b.active = False
                    if b.owner == "player" and not b.predicted_bunker_hit:
                        self.reward_adjustment -= PLAYER_BUNKER_HIT_PENALTY
                        self.player_bunker_hits += 1

        # Nepřátelé vs bunkry (dotyk)
        for bunker in self.bunkers:
            for e in self.formation.alive_enemies():
                if e.rect.colliderect(
                        pygame.Rect(bunker.x, bunker.y,
                                    22 * BUNKER_BLOCK_SIZE,
                                    16 * BUNKER_BLOCK_SIZE)):
                    bunker.damage_area(e.x + ENEMY_WIDTH // 2,
                                       e.y + ENEMY_HEIGHT // 2, radius=20)

        # Vyčištění neaktivních střel
        self.player_bullets = [b for b in self.player_bullets if b.active]
        self.enemy_bullets = [b for b in self.enemy_bullets if b.active]

    def _try_player_shoot(self):
        if not self._can_player_shoot():
            return False

        bx = self.player.x + self.player.width // 2 - BULLET_WIDTH // 2
        by = self.player.y
        bullet = Bullet(bx, by, -BULLET_SPEED, color=GREEN, owner="player")
        if self._shot_path_hits_bunker(bx):
            bullet.predicted_bunker_hit = True
            self.reward_adjustment -= PLAYER_BLOCKED_SHOT_PENALTY
            self.player_blocked_shots += 1
        elif self._shot_path_has_target(bx):
            self.reward_adjustment += PLAYER_TARGETED_SHOT_BONUS
            self.player_targeted_shots += 1

        self.player_bullets.append(bullet)
        self.player.last_shot_frame = self.frame_count
        self.player_shots_fired += 1
        return True

    def _can_player_shoot(self):
        if len(self.player_bullets) >= MAX_OBS_PLAYER_BULLETS:
            return False

        return (
            self.frame_count - self.player.last_shot_frame
            >= BULLET_COOLDOWN_FRAMES
        )

    def _get_player_bullet_x(self):
        return self.player.x + self.player.width // 2 - BULLET_WIDTH // 2

    def _has_clear_shot_from_position(self, bullet_x):
        return (
            not self._shot_path_hits_bunker(bullet_x)
            and self._shot_path_has_target(bullet_x)
        )

    def _get_nearest_clear_shot_x(self, current_bullet_x):
        candidate_xs = []

        for enemy in self.formation.enemies:
            if not enemy.alive:
                continue
            bullet_x = enemy.x + ENEMY_WIDTH // 2 - BULLET_WIDTH // 2
            if not self._shot_path_hits_bunker(bullet_x):
                candidate_xs.append(bullet_x)

        if self.mystery.active:
            bullet_x = self.mystery.x + self.mystery.width // 2 - BULLET_WIDTH // 2
            if not self._shot_path_hits_bunker(bullet_x):
                candidate_xs.append(bullet_x)

        if not candidate_xs:
            return None

        return min(candidate_xs, key=lambda x: abs(x - current_bullet_x))

    def _reward_movement_toward_clear_shot(
        self,
        old_bullet_x,
        nearest_clear_lane,
        had_clear_shot,
    ):
        if had_clear_shot or nearest_clear_lane is None:
            return

        new_bullet_x = self._get_player_bullet_x()
        old_distance = abs(old_bullet_x - nearest_clear_lane)
        new_distance = abs(new_bullet_x - nearest_clear_lane)

        if new_distance < old_distance:
            self.reward_adjustment += MOVE_TOWARD_CLEAR_SHOT_BONUS
        elif new_distance > old_distance:
            self.reward_adjustment -= MOVE_AWAY_FROM_CLEAR_SHOT_PENALTY

    def _apply_idle_clear_shot_penalty(self, idle_action):
        if not idle_action:
            return

        if not self._can_player_shoot():
            return

        if self.player_bullets:
            return

        if self._has_clear_shot_from_position(self._get_player_bullet_x()):
            self.reward_adjustment -= IDLE_WITH_CLEAR_SHOT_PENALTY

    def _shot_path_hits_bunker(self, bullet_x):
        shot_lane = pygame.Rect(bullet_x, 0, BULLET_WIDTH, self.player.y)
        for bunker in self.bunkers:
            for block in bunker.blocks:
                if block.colliderect(shot_lane):
                    return True
        return False

    def _shot_path_has_target(self, bullet_x):
        bullet_left = bullet_x
        bullet_right = bullet_x + BULLET_WIDTH

        for enemy in self.formation.enemies:
            if enemy.alive and enemy.rect.right >= bullet_left:
                if enemy.rect.left <= bullet_right:
                    return True

        if self.mystery.active and self.mystery.rect.right >= bullet_left:
            if self.mystery.rect.left <= bullet_right:
                return True

        return False

    def _tick_timers(self):
        if self.death_timer > 0:
            self.death_timer -= 1

    def _check_win_lose(self):
        # Hráč ztratil všechny životy
        if self.lives <= 0:
            self.game_over = True
            return

        # Nepřátelé dosáhli úrovně hráče
        if self.formation.lowest_y() >= PLAYER_Y:
            self.game_over = True
            return

        # Všichni nepřátelé zničeni – další level
        if self.formation.alive_count() == 0:
            if self.single_level_mode:
                self.victory = True
                return
            self._next_level()

    def _next_level(self):
        self.level += 1
        self.formation = EnemyFormation()
        self.player_bullets.clear()
        self.enemy_bullets.clear()
        self.mystery = MysteryShip(self.rng)
        self.enemy_shoot_chance = min(0.008, ENEMY_SHOOT_CHANCE + self.level * 0.001)

    # -------------------------------------------------------------------
    # Vykreslování
    # -------------------------------------------------------------------

    def _draw(self):
        self.screen.fill(BLACK)

        # Bunkry
        for bunker in self.bunkers:
            bunker.draw(self.screen)

        # Hráč (bliká při zásahu)
        if self.death_timer <= 0 or self.death_timer % 6 < 3:
            self.player.draw(self.screen)

        # Nepřátelé
        self.formation.draw(self.screen)

        # Mystery ship
        self.mystery.draw(self.screen)

        # Střely
        for b in self.player_bullets:
            b.draw(self.screen)
        for b in self.enemy_bullets:
            b.draw(self.screen)

        # Score popups
        new_popups = []
        for text, x, y, timer in self.score_popups:
            if timer > 0:
                surf = self.font.render(text, True, WHITE)
                self.screen.blit(surf, (x, y - (30 - timer)))
                new_popups.append((text, x, y, timer - 1))
        self.score_popups = new_popups

        # HUD
        score_text = self.font.render(f"SCORE: {self.score:05d}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

        lives_text = self.font.render(f"LIVES: {self.lives}", True, GREEN)
        self.screen.blit(lives_text, (SCREEN_WIDTH - 150, 10))

        level_text = self.font.render(f"LEVEL: {self.level}", True, CYAN)
        self.screen.blit(level_text, (SCREEN_WIDTH // 2 - 50, 10))

        # Čára na spodku
        pygame.draw.line(self.screen, GREEN,
                         (0, SCREEN_HEIGHT - 30),
                         (SCREEN_WIDTH, SCREEN_HEIGHT - 30), 2)

        # Ikony životů
        for i in range(self.lives - 1):
            draw_player(self.screen,
                        SCREEN_WIDTH - 140 + i * 50,
                        SCREEN_HEIGHT - 25, color=GREEN)

    def _draw_game_over(self):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        text = self.big_font.render("GAME OVER", True, RED)
        rect = text.get_rect(center=(SCREEN_WIDTH // 2,
                                     SCREEN_HEIGHT // 2 - 40))
        self.screen.blit(text, rect)

        score_text = self.font.render(f"FINAL SCORE: {self.score:05d}",
                                      True, WHITE)
        sr = score_text.get_rect(center=(SCREEN_WIDTH // 2,
                                         SCREEN_HEIGHT // 2 + 20))
        self.screen.blit(score_text, sr)

        restart = self.font.render("Press R to restart or ESC to quit",
                                   True, YELLOW)
        rr = restart.get_rect(center=(SCREEN_WIDTH // 2,
                                      SCREEN_HEIGHT // 2 + 60))
        self.screen.blit(restart, rr)

    def _draw_start_screen(self):
        self.screen.fill(BLACK)

        title = self.big_font.render("SPACE INVADERS", True, GREEN)
        tr = title.get_rect(center=(SCREEN_WIDTH // 2, 120))
        self.screen.blit(title, tr)

        instructions = [
            "Arrow Keys / A,D - Move",
            "Space - Shoot",
            "P - Pause",
            "ESC - Quit",
            "",
            "= SCORE TABLE =",
            "",
        ]

        y = 220
        for line in instructions:
            t = self.font.render(line, True, WHITE)
            r = t.get_rect(center=(SCREEN_WIDTH // 2, y))
            self.screen.blit(t, r)
            y += 28

        # Ukázkový nepřítel s body
        demo_enemies = [
            (draw_enemy_type_a, MAGENTA, "= 30 PTS"),
            (draw_enemy_type_b, CYAN, "= 20 PTS"),
            (draw_enemy_type_c, WHITE, "= 10 PTS"),
        ]
        for func, color, pts in demo_enemies:
            func(self.screen, SCREEN_WIDTH // 2 - 80, y - 6, color, 0)
            t = self.font.render(pts, True, WHITE)
            self.screen.blit(t, (SCREEN_WIDTH // 2 - 40, y))
            y += 36

        # Mystery ship
        pygame.draw.rect(self.screen, RED,
                         (SCREEN_WIDTH // 2 - 80, y, 32, 12))
        t = self.font.render("= ??? PTS", True, WHITE)
        self.screen.blit(t, (SCREEN_WIDTH // 2 - 40, y - 2))
        y += 50

        start = self.font.render("Press ENTER to start", True, YELLOW)
        sr = start.get_rect(center=(SCREEN_WIDTH // 2, y))
        self.screen.blit(start, sr)

    # -------------------------------------------------------------------
    # Hlavní herní smyčka (pro lidského hráče)
    # -------------------------------------------------------------------

    def run(self):
        """Spustí hru s ovládáním klávesnicí."""
        state = "start"  # start, playing, paused, game_over
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if state == "paused":
                            state = "playing"
                        else:
                            running = False

                    if state == "start":
                        if event.key == pygame.K_RETURN:
                            self.reset()
                            state = "playing"

                    elif state == "playing":
                        if event.key == pygame.K_p:
                            state = "paused"

                    elif state == "paused":
                        if event.key == pygame.K_p:
                            state = "playing"

                    elif state == "game_over":
                        if event.key == pygame.K_r:
                            self.reset()
                            state = "playing"

            if state == "start":
                self._draw_start_screen()

            elif state == "playing":
                self._game_tick()
                self._draw()
                if self.game_over:
                    state = "game_over"

            elif state == "paused":
                self._draw()
                # Zobrazení textu pauzy
                pause_text = self.big_font.render("PAUSED", True, YELLOW)
                pr = pause_text.get_rect(
                    center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
                self.screen.blit(pause_text, pr)

            elif state == "game_over":
                self._draw()
                self._draw_game_over()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

    def _game_tick(self):
        """Jeden frame herní logiky při lidském ovládání."""
        self.frame_count += 1
        keys = pygame.key.get_pressed()

        # Pohyb
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.player.move(-1)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.player.move(1)

        # Střelba
        if keys[pygame.K_SPACE]:
            self._try_player_shoot()

        # Aktualizace
        self._update_bullets()
        self.formation.update()
        self.mystery.update()
        self._enemy_shoot()
        self._check_collisions()
        self._tick_timers()
        self._check_win_lose()

    def close(self):
        pygame.quit()


# ---------------------------------------------------------------------------
# Spuštění
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    game = SpaceInvadersGame(human_mode=True)
    game.run()
