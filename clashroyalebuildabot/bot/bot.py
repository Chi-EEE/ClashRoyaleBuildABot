import time

from loguru import logger

from clashroyalebuildabot.bot.action import Action
from clashroyalebuildabot.data.constants import (
    ALLY_TILES,
    LEFT_PRINCESS_TILES,
    RIGHT_PRINCESS_TILES,
    TILE_HEIGHT,
    TILE_WIDTH,
    DISPLAY_CARD_WIDTH,
    DISPLAY_CARD_HEIGHT,
    DISPLAY_CARD_Y,
    DISPLAY_CARD_INIT_X,
    DISPLAY_CARD_DELTA_X,
    SCREEN_CONFIG,
    TILE_INIT_X,
    TILE_INIT_Y,
    DISPLAY_HEIGHT,
)
from clashroyalebuildabot.screen import Screen
from clashroyalebuildabot.state.detector import Detector


class Bot:
    def __init__(
        self, card_names, action_class=Action, auto_start=True, debug=False
    ):
        self.card_names = card_names
        self.action_class = action_class
        self.auto_start = auto_start
        self.debug = debug

        self.screen = Screen()
        self.detector = Detector(card_names, debug=self.debug)
        self.state = None

    @staticmethod
    def _get_nearest_tile(x, y):
        tile_x = round(((x - TILE_INIT_X) / TILE_WIDTH) - 0.5)
        tile_y = round(
            ((DISPLAY_HEIGHT - TILE_INIT_Y - y) / TILE_HEIGHT) - 0.5
        )
        return tile_x, tile_y

    @staticmethod
    def _get_tile_centre(tile_x, tile_y):
        x = TILE_INIT_X + (tile_x + 0.5) * TILE_WIDTH
        y = DISPLAY_HEIGHT - TILE_INIT_Y - (tile_y + 0.5) * TILE_HEIGHT
        return x, y

    @staticmethod
    def _get_card_centre(card_n):
        x = (
            DISPLAY_CARD_INIT_X
            + DISPLAY_CARD_WIDTH / 2
            + card_n * DISPLAY_CARD_DELTA_X
        )
        y = DISPLAY_CARD_Y + DISPLAY_CARD_HEIGHT / 2
        return x, y

    def _get_valid_tiles(self):
        tiles = ALLY_TILES
        if self.state["numbers"]["left_enemy_princess_hp"]["number"] == 0:
            tiles += LEFT_PRINCESS_TILES
        if self.state["numbers"]["right_enemy_princess_hp"]["number"] == 0:
            tiles += RIGHT_PRINCESS_TILES
        return tiles

    def get_actions(self):
        if len(self.state) == 0:
            return []
        all_tiles = ALLY_TILES + LEFT_PRINCESS_TILES + RIGHT_PRINCESS_TILES
        valid_tiles = self._get_valid_tiles()

        actions = []
        for i in range(4):
            card = self.state["cards"][i + 1]
            enough_elixir = (
                int(self.state["numbers"]["elixir"]["number"]) >= card["cost"]
            )
            ready = card["ready"]
            not_blank = card["name"] != "blank"
            if enough_elixir and ready and not_blank:
                if card["type"] == "spell":
                    tiles = all_tiles
                else:
                    tiles = valid_tiles
                actions.extend(
                    [
                        self.action_class(i, x, y, *card.values())
                        for (x, y) in tiles
                    ]
                )

        return actions

    def set_state(self):
        try:
            screenshot = self.screen.take_screenshot()
            self.state = self.detector.run(screenshot)
            if self.auto_start:
                if self.state["screen"] != "in_game":
                    self.screen.click(
                        *SCREEN_CONFIG[self.state["screen"]][
                            "click_coordinates"
                        ]
                    )
                    time.sleep(2)

        except Exception as e:  # Catch any exception from take_screenshot
            logger.error(f"Error occurred while taking screenshot: {e}")
            # You might want to add additional error handling or recovery logic here

    def play_action(self, action):
        card_centre = self._get_card_centre(action.index)
        tile_centre = self._get_tile_centre(action.tile_x, action.tile_y)
        self.screen.click(*card_centre)
        self.screen.click(*tile_centre)
