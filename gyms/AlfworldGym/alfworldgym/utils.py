import re
from typing import Dict, Any, Optional, Tuple

def validate_action_format(action_input: str) -> Tuple[bool, str]:
    if not action_input or not isinstance(action_input, str):
        return False, "Action must be a non-empty string"
    
    action_input = action_input.strip()
    
    if not (action_input.startswith("[action]") or action_input.startswith("[finish]")):
        return False, "You must give an action through the tool call"
    
    if action_input.startswith("[action]"):
        content = action_input[8:].strip()
        # further validate its format to be one of the following:
        # """
        # Available commands:
        # look:                             look around your current location
        # inventory:                        check your current inventory
        # go to (receptacle):               move to a receptacle
        # open (receptacle):                open a receptacle
        # close (receptacle):               close a receptacle
        # take (object) from (receptacle):  take an object from a receptacle
        # move (object) to (receptacle):    place an object in or on a receptacle
        # examine (object):                 examine an object in detail
        # use (object):                     use an object
        # heat (object) with (receptacle):  heat an object using a receptacle
        # clean (object) with (receptacle): clean an object using a receptacle
        # cool (object) with (receptacle):  cool an object using a receptacle
        # slice (object) with (object):     slice an object using a sharp object
        # """
        # Please align the patterns with the actual commands used in your environment
        valid_commands = [
            r"look",
            r"inventory",
            r"help",
            r"go to \w+",
            r"open \w+",
            r"close \w+",
            r"take \w+ from \w+",
            r"move \w+ to \w+",
            r"examine \w+",
            r"use \w+",
            r"heat \w+ with \w+",
            r"clean \w+ with \w+",
            r"cool \w+ with \w+",
            r"slice \w+ with \w+"
        ]

        # Check if the action matches any valid command
        for command in valid_commands:
            if re.match(command, content):
                return True, ""

        return False, f"{content}. Please follow the specified action format in instruction or choose from admissible commands in your action content."

