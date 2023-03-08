import fasttext
from logger import logger


if __name__ == "__main__":
    model = fasttext.train_supervised(input="./data/train.txt", epoch=100, lr=1e0)
    result = model.test("./data/test.txt")
    logger.info(result)