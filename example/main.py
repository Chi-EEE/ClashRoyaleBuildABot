"""
A CustomBot implementation through import
"""
from two_six_hog_cycle_bot import TwoSixHogCycle # see custom_bot.py


def main():
    # Set required bot variables
    card_names = ['hog_rider', 'the_log', 'fireball', 'ice_spirit', 'ice_golem', 'skeletons', 'cannon', 'musketeer']
    # Define an instance of CustomBot
    bot = TwoSixHogCycle(card_names, debug=True)
    # and run!
    bot.run()


if __name__ == '__main__':
    main()