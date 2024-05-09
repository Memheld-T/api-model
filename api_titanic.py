import joblib
import uvicorn
from fastapi import FastAPI
import pandas as pd
from prometheus_client import make_asgi_app, Counter

app = FastAPI()
metrics_app = make_asgi_app()
app.mount("/metrics",metrics_app)

survived_counter=Counter("survived","Number of survived passengers")
not_survived_counter=Counter("not_survived","Number of not survived passengers")

# Load model
titanic_model = joblib.load("model_titanic.joblib")

@app.post("/titanic")
def prediction_api(pclass: int, sex: int, age: int) -> bool:

    # predict
    x = [pclass, sex, age]
    prediction = titanic_model.predict(pd.DataFrame(x).transpose())

    survived=prediction[0] == 1
    if survived:
        survived_counter.inc()
    else:
        not_survived_counter.inc()

    return prediction[0] == 1


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
