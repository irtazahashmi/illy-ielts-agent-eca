# IELTS Agent Illy

## Starting the AI

0. Install rust compiler
1. Install all libraries from the `requirements.txt`
   `python -m pip install -r requirements.txt`
2. Install the `spacy` pre-trained models
```python -m spacy download en```
3. Run Furhat SDK Desktop Launcher
4. Start Remote API
   > Start the database if you haven't done that already
5. Run `run.py`

## Database setup

1. Install MongoDB on your machine
2. Run it before starting the agent

## Running Tests

To run the test simply run
`./test.sh`
or make sure you run `pytest ./test` with the environment variable `PYTHONPATH=./src`
