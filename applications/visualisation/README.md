## Visualisation Code

This code is to visualise a given instance of the `ShortDeckPokerState`.

It looks like this:
<p align="center">
  <img src="https://github.com/fedden/pluribus-poker-AI/blob/develop/assets/visualisation.png">
</p>

### How to run

First build the frontend, this will be served a static files by the `run.py` script.
```bash
cd frontend
npm run build
```

Next run the backend.
```bash
cd .. 
FLASK_APP=run.py python -m flask run 
```

### More to do

TODO:
* Fix the community cards (not showing).
* Improve the "..." icon that hovers above the currently playing player.
* Generalise the code so we can hook up this library to an python API and visualise any state instance, like TensorBoard does.
* Dockerise this for ease of use.
