from config import Config
from data import load_data, train_val_split, normalize_features
from model import LinearRegression
from train import train

config = Config()

def main():
    data = load_data(config.DATA_PATH)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_split(data)

    print(X_train.shape, y_train.shape)

    model = LinearRegression(num_features=X_train.shape[1])

    train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=Config.NUM_EPOCHS,
        lr=Config.LEARNING_RATE,
    )

if __name__ == "__main__":
    main()

