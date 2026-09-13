# Browser Python runner

Turn on **Enable Python runner** below **Show reasoning** to review and run code; it is off by default.
**Load code from chat** refreshes the snippets. Select one by its code preview, then click **Run Python**.
Script input appears for code using `input()`. Empty input supplies a blank answer; **Stop** cancels a run.
The first run downloads [Pyodide](https://pyodide.org/); supported imported packages load automatically.
Runs use a temporary browser filesystem, with a 60-second execution limit and capped output.
Desktop apps, GPU packages and some network requests require a separate local Python environment.
