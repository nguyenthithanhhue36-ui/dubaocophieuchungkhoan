# dubaocophieuchungkhoan
future_predictions = []
last_60_days = last_60_days.copy()

for i in range(forecast_days):
    next_pred = model.predict(last_60_days, verbose=0)
    future_predictions.append(next_pred[0, 0])

    last_60_days = np.concatenate(
        [
            last_60_days[:, 1:, :],
            next_pred.reshape(1, 1, 1)
        ],
        axis=1
    )

future_predictions = np.array(
    future_predictions
).reshape(-1, 1)

future_predictions = scaler.inverse_transform(
    future_predictions
)
