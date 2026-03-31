"""
Space Invaders game logic with an RL-friendly state representation.

The same core game can be driven either by keyboard input or by an agent.
The important design goal is that the current game state can be exported as
one flat numeric vector for neural-network training.
"""

import math
import random
import pygame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
ORANGE = (255, 165, 0)

# Player
PLAYER_WIDTH = 48
PLAYER_HEIGHT = 32
PLAYER_SPEED = 5
PLAYER_Y = SCREEN_HEIGHT - 60

# Player bullets
BULLET_WIDTH = 4
BULLET_HEIGHT = 12
BULLET_SPEED = 8
BULLET_COOLDOWN_MS = 300  # milliseconds
BULLET_COOLDOWN_FRAMES = max(1, round(BULLET_COOLDOWN_MS * FPS / 1000))
PLAYER_INVULNERABILITY_FRAMES = 90

# Enemies
ENEMY_ROWS = 5
ENEMY_COLS = 11
ENEMY_WIDTH = 32
ENEMY_HEIGHT = 24
ENEMY_PADDING_X = 16
ENEMY_PADDING_Y = 12
ENEMY_START_X = 60
ENEMY_START_Y = 80
ENEMY_BASE_SPEED = 1.0
ENEMY_SPEED_INCREASE = 0.15  # kept for balancing experiments
ENEMY_DROP = 16  # pixels the formation drops after hitting a wall

# Enemy bullets
ENEMY_BULLET_WIDTH = 4
ENEMY_BULLET_HEIGHT = 12
ENEMY_BULLET_SPEED = 4
ENEMY_SHOOT_CHANCE = 0.002  # per-frame shoot chance for each active column

# Mystery ship
MYSTERY_WIDTH = 48
MYSTERY_HEIGHT = 20
MYSTERY_SPEED = 3
MYSTERY_SCORES = [50, 100, 150, 300]
MYSTERY_INTERVAL = 600  # frames between mystery ship spawns

# Bunkers (shields)
BUNKER_COUNT = 4
BUNKER_BLOCK_SIZE = 4
BUNKER_Y = SCREEN_HEIGHT - 140
MAX_OBS_ENEMY_BULLETS = 8
MAX_OBS_PLAYER_BULLETS = 3
MAX_TRACKED_FRAMES = 5000

# Reward shaping used by the RL step() interface
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

# Score per row, from the bottom row to the top row
ROW_SCORES = [10, 10, 20, 20, 30]

# ---------------------------------------------------------------------------
# Helper functions for pixel-art rendering
# ---------------------------------------------------------------------------

def draw_player(surface, x, y, color=GREEN):
    """Draw the player cannon using the classic arcade silhouette."""
    # Main body
    pygame.draw.rect(surface, color, (x + 4, y + 12, 40, 20))
    # Barrel
    pygame.draw.rect(surface, color, (x + 20, y, 8, 16))
    # Base
    pygame.draw.rect(surface, color, (x, y + 24, 48, 8))


def draw_enemy_type_a(surface, x, y, color, frame):
    """Draw the small top-row enemy variant."""
    w, h = 24, 18
    ox = x + (ENEMY_WIDTH - w) // 2
    oy = y + (ENEMY_HEIGHT - h) // 2
    # Body
    pygame.draw.rect(surface, color, (ox + 8, oy, 8, 6))
    pygame.draw.rect(surface, color, (ox + 4, oy + 6, 16, 6))
    pygame.draw.rect(surface, color, (ox, oy + 12, 24, 6))
    # Eyes
    pygame.draw.rect(surface, BLACK, (ox + 8, oy + 12, 4, 4))
    pygame.draw.rect(surface, BLACK, (ox + 14, oy + 12, 4, 4))
    # Toggle the feet position to emulate the original two-frame animation.
    if frame == 0:
        pygame.draw.rect(surface, color, (ox + 2, oy + 18, 4, 4))
        pygame.draw.rect(surface, color, (ox + 18, oy + 18, 4, 4))
    else:
        pygame.draw.rect(surface, color, (ox - 2, oy + 16, 4, 4))
        pygame.draw.rect(surface, color, (ox + 22, oy + 16, 4, 4))


def draw_enemy_type_b(surface, x, y, color, frame):
    """Draw the middle enemy variant used by the center rows."""
    w, h = 28, 20
    ox = x + (ENEMY_WIDTH - w) // 2
    oy = y + (ENEMY_HEIGHT - h) // 2
    # Body
    pygame.draw.rect(surface, color, (ox + 4, oy, 20, 8))
    pygame.draw.rect(surface, color, (ox, oy + 8, 28, 8))
    # Eyes
    pygame.draw.rect(surface, BLACK, (ox + 6, oy + 8, 4, 4))
    pygame.draw.rect(surface, BLACK, (ox + 18, oy + 8, 4, 4))
    # Lower body
    pygame.draw.rect(surface, color, (ox + 4, oy + 16, 8, 4))
    pygame.draw.rect(surface, color, (ox + 16, oy + 16, 8, 4))
    if frame == 0:
        pygame.draw.rect(surface, color, (ox, oy + 14, 4, 4))
        pygame.draw.rect(surface, color, (ox + 24, oy + 14, 4, 4))
    else:
        pygame.draw.rect(surface, color, (ox - 2, oy + 16, 4, 4))
        pygame.draw.rect(surface, color, (ox + 26, oy + 16, 4, 4))


def draw_enemy_type_c(surface, x, y, color, frame):
    """Draw the large enemy variant used by the bottom rows."""
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
    """Return a 2D mask describing the default bunker shape."""
    # The mask keeps the classic arch on top and a notch in the center bottom.
    cols = 22
    rows = 16
    shape = []
    for r in range(rows):
        row = []
        for c in range(cols):
            # Carve the rounded top edge.
            cx = cols / 2
            cy = 4
            dist = math.sqrt((c - cx) ** 2 + (r - cy) ** 2)
            if r < 4 and dist > 10:
                row.append(0)
            elif r >= 12 and 8 <= c <= 13:
                # Leave a firing opening below the bunker.
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
# Game objects
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
        self.direction = 1  # 1 = moving right, -1 = moving left
        self.speed = ENEMY_BASE_SPEED
        self.move_timer = 0
        self.move_interval = 40  # frames between formation moves
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

        # The classic game speeds up as the formation thins out.
        total = ENEMY_ROWS * ENEMY_COLS
        killed = total - len(alive)
        self.move_interval = max(2, 40 - killed)

        if self.move_timer >= self.move_interval:
            self.move_timer = 0
            self.anim_frame = 1 - self.anim_frame

            # Decide whether the next horizontal step would hit a screen edge.
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
        """Return the lowest living enemy in each column."""
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
        """Remove blocks around an impact point to create a rough explosion hole."""
        self.blocks = [b for b in self.blocks
                       if math.sqrt((b.centerx - cx) ** 2 +
                                    (b.centery - cy) ** 2) > radius]

    def draw(self, surface):
        for b in self.blocks:
            pygame.draw.rect(surface, GREEN, b)


# ---------------------------------------------------------------------------
# Main game class
# ---------------------------------------------------------------------------

class SpaceInvadersGame:
    """
    Own the full game state for both human play and RL training.

    Human mode creates a real window and runs the keyboard loop. Headless mode
    renders into an off-screen surface and exposes step() for agent control.
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
            # In headless training we only need font support. Avoiding full
            # pygame.init() keeps subprocess workers away from GUI setup.
            if not pygame.font.get_init():
                pygame.font.init()

        if human_mode:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Space Invaders")
        else:
            # Headless mode still draws into a surface so rendering code stays shared.
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
        """Reset the game to its initial state and return the first observation."""
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
    # AI / neural network interface
    # -------------------------------------------------------------------

    def get_state_vector(self):
        """
        Return the current state as one flat float vector.

        Most values are normalized to 0..1. The vector includes player state,
        formation state, the mystery ship, a fixed number of projectile slots,
        and bunker health. A fixed layout is important so the agent always sees
        the same observation shape.
        """
        state = []
        cooldown_remaining = max(
            0,
            BULLET_COOLDOWN_FRAMES
            - (self.frame_count - self.player.last_shot_frame),
        )

        # Player state
        state.append(self.player.x / SCREEN_WIDTH)
        state.append(self.lives / 3.0)
        state.append(min(self.score / 10000.0, 1.0))
        state.append(len(self.player_bullets) / MAX_OBS_PLAYER_BULLETS)
        state.append(cooldown_remaining / BULLET_COOLDOWN_FRAMES)
        state.append(self.death_timer / PLAYER_INVULNERABILITY_FRAMES)
        state.append(min(self.frame_count / MAX_TRACKED_FRAMES, 1.0))

        # Encode each enemy in a stable order so the network can learn column
        # and row structure instead of seeing a shuffled list every frame.
        for e in self.formation.enemies:
            state.append(1.0 if e.alive else 0.0)
        for e in self.formation.enemies:
            state.append(e.x / SCREEN_WIDTH)
        for e in self.formation.enemies:
            state.append(e.y / SCREEN_HEIGHT)

        # Formation
        state.append(1.0 if self.formation.direction > 0 else 0.0)

        # Mystery ship
        state.append(1.0 if self.mystery.active else 0.0)
        state.append(self.mystery.x / SCREEN_WIDTH if self.mystery.active
                     else 0.0)

        # Track the most urgent enemy bullets first: the lowest ones are the
        # closest threats to the player.
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

        # Track player bullets from top to bottom because higher bullets are
        # closer to resolving into a hit or a miss.
        tracked_player_bullets = sorted(self.player_bullets, key=lambda b: b.y)
        for i in range(MAX_OBS_PLAYER_BULLETS):
            if i < len(tracked_player_bullets):
                b = tracked_player_bullets[i]
                state.append(b.x / SCREEN_WIDTH)
                state.append(b.y / SCREEN_HEIGHT)
            else:
                state.append(0.0)
                state.append(0.0)

        # Bunkers
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
        Advance the game by one agent action.

        Action mapping:
            0 = idle
            1 = move left
            2 = move right
            3 = shoot
            4 = move left + shoot
            5 = move right + shoot

        Return: (state_vector, reward, done, info)
        """
        self.frame_count += 1
        self.reward_adjustment = 0.0

        # Snapshot the previous frame so the reward can measure improvement
        # instead of relying on absolute values only.
        old_score = self.score
        old_lives = self.lives
        old_bullet_x = self._get_player_bullet_x()
        had_clear_shot = self._has_clear_shot_from_position(old_bullet_x)
        nearest_clear_lane = self._get_nearest_clear_shot_x(old_bullet_x)

        # Translate the compact discrete action into movement and shooting flags.
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

        # Reward movement only when it helps line up a currently blocked shot.
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

        # Fire after movement so combined actions use the new lane immediately.
        if shoot:
            self._try_player_shoot()

        # Run one full game-tick worth of simulation.
        self._update_bullets()
        self.formation.update()
        self.mystery.update()
        self._enemy_shoot()
        self._check_collisions()
        self._tick_timers()
        self._check_win_lose()

        # Score delta is the primary signal. Small shaped terms help the agent
        # discover useful behavior before it consistently wins levels.
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
    # Game logic
    # -------------------------------------------------------------------

    def _update_bullets(self):
        for b in self.player_bullets:
            b.update()
            if not b.active and not b.hit_target:
                # A player bullet that leaves the screen without hitting
                # anything is treated as a wasted shot.
                self.reward_adjustment -= PLAYER_MISSED_SHOT_PENALTY
                self.player_shots_missed += 1
        for b in self.enemy_bullets:
            b.update()
        self.player_bullets = [b for b in self.player_bullets if b.active]
        self.enemy_bullets = [b for b in self.enemy_bullets if b.active]

    def _enemy_shoot(self):
        # Only the lowest living enemy in each column can shoot, matching the
        # original arcade rules and preventing bullets from spawning inside allies.
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
        # Player bullets vs enemies
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

        # Player bullets vs mystery ship
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

        # Enemy bullets vs player
        if self.death_timer <= 0:
            for b in self.enemy_bullets:
                if not b.active:
                    continue
                if b.rect.colliderect(self.player.rect):
                    b.active = False
                    self.lives -= 1
                    self.death_timer = PLAYER_INVULNERABILITY_FRAMES
                    break

        # Bullets vs bunkers
        for bunker in self.bunkers:
            for b in self.player_bullets + self.enemy_bullets:
                if not b.active:
                    continue
                if bunker.check_collision(b.rect):
                    bunker.damage_area(b.x + b.width // 2,
                                       b.y + b.height // 2)
                    b.active = False
                    # Predicted bunker hits were already penalized on fire.
                    # This branch catches unexpected bunker collisions after movement.
                    if b.owner == "player" and not b.predicted_bunker_hit:
                        self.reward_adjustment -= PLAYER_BUNKER_HIT_PENALTY
                        self.player_bunker_hits += 1

        # Enemies touching a bunker chew through a larger area than bullets do.
        for bunker in self.bunkers:
            for e in self.formation.alive_enemies():
                if e.rect.colliderect(
                        pygame.Rect(bunker.x, bunker.y,
                                    22 * BUNKER_BLOCK_SIZE,
                                    16 * BUNKER_BLOCK_SIZE)):
                    bunker.damage_area(e.x + ENEMY_WIDTH // 2,
                                       e.y + ENEMY_HEIGHT // 2, radius=20)

        # Remove bullets that were deactivated by a collision this frame.
        self.player_bullets = [b for b in self.player_bullets if b.active]
        self.enemy_bullets = [b for b in self.enemy_bullets if b.active]

    def _try_player_shoot(self):
        if not self._can_player_shoot():
            return False

        bx = self.player.x + self.player.width // 2 - BULLET_WIDTH // 2
        by = self.player.y
        bullet = Bullet(bx, by, -BULLET_SPEED, color=GREEN, owner="player")
        if self._shot_path_hits_bunker(bx):
            # Mark obviously blocked shots up front so the agent learns that
            # firing into its own bunker is different from a clean lane shot.
            bullet.predicted_bunker_hit = True
            self.reward_adjustment -= PLAYER_BLOCKED_SHOT_PENALTY
            self.player_blocked_shots += 1
        elif self._shot_path_has_target(bx):
            # A clear lane with an enemy or mystery ship above is slightly encouraged.
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

        # Use the closest valid firing lane as the movement shaping target.
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

        # Penalize hesitation only when the player is ready to shoot and already
        # has a clean lane. This avoids pushing random spam during cooldowns.
        if self._has_clear_shot_from_position(self._get_player_bullet_x()):
            self.reward_adjustment -= IDLE_WITH_CLEAR_SHOT_PENALTY

    def _shot_path_hits_bunker(self, bullet_x):
        # Model a full vertical shot lane from the player upward to answer
        # "would this shot be blocked immediately?" without firing first.
        shot_lane = pygame.Rect(bullet_x, 0, BULLET_WIDTH, self.player.y)
        for bunker in self.bunkers:
            for block in bunker.blocks:
                if block.colliderect(shot_lane):
                    return True
        return False

    def _shot_path_has_target(self, bullet_x):
        bullet_left = bullet_x
        bullet_right = bullet_x + BULLET_WIDTH

        # A target is considered hittable when its horizontal span overlaps
        # the vertical shot lane.
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
        # The player has no lives left.
        if self.lives <= 0:
            self.game_over = True
            return

        # The formation has reached the player line.
        if self.formation.lowest_y() >= PLAYER_Y:
            self.game_over = True
            return

        # Clearing the formation either ends the episode or starts the next level.
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
        # Difficulty ramps up by increasing the enemy firing chance each level.
        self.enemy_shoot_chance = min(0.008, ENEMY_SHOOT_CHANCE + self.level * 0.001)

    # -------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------

    def _draw(self):
        self.screen.fill(BLACK)

        # Bunkers
        for bunker in self.bunkers:
            bunker.draw(self.screen)

        # Flash the player briefly after a hit to show invulnerability frames.
        if self.death_timer <= 0 or self.death_timer % 6 < 3:
            self.player.draw(self.screen)

        # Enemies
        self.formation.draw(self.screen)

        # Mystery ship
        self.mystery.draw(self.screen)

        # Bullets
        for b in self.player_bullets:
            b.draw(self.screen)
        for b in self.enemy_bullets:
            b.draw(self.screen)

        # Floating score popups fade out over a fixed number of frames.
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

        # Ground line
        pygame.draw.line(self.screen, GREEN,
                         (0, SCREEN_HEIGHT - 30),
                         (SCREEN_WIDTH, SCREEN_HEIGHT - 30), 2)

        # Life icons
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

        # Show one sample sprite for each score tier.
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
    # Main loop for human play
    # -------------------------------------------------------------------

    def run(self):
        """Run the interactive keyboard-controlled game loop."""
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
                # Draw the pause banner on top of the current frame.
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
        """Advance one frame of game logic for keyboard-controlled play."""
        self.frame_count += 1
        keys = pygame.key.get_pressed()

        # Movement
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.player.move(-1)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.player.move(1)

        # Shooting
        if keys[pygame.K_SPACE]:
            self._try_player_shoot()

        # Regular world update
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
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    game = SpaceInvadersGame(human_mode=True)
    game.run()
