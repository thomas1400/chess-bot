import os
import sys

def init():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
