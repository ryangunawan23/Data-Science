{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "dress-serbia",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "optimum-introduction",
   "metadata": {},
   "outputs": [],
   "source": [
    "input_data = [\n",
    "    [0, 0],\n",
    "    [0, 1],\n",
    "    [1, 0],\n",
    "    [1, 1]\n",
    "]\n",
    "\n",
    "target = [0, 1, 1, 1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "opposed-probe",
   "metadata": {},
   "outputs": [],
   "source": [
    "weight = np.random.normal(size=(2,1))\n",
    "bias = np.random.normal(size = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "complete-session",
   "metadata": {},
   "outputs": [],
   "source": [
    "def feedForward(inputs):\n",
    "    value = np.matmul(inputs, weight)\n",
    "    value += bias\n",
    "    return activationFunc(value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "cooked-knowing",
   "metadata": {},
   "outputs": [],
   "source": [
    "def activationFunc(x):\n",
    "    if x>=0:\n",
    "        return 1\n",
    "    else:\n",
    "        return 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "reflected-publication",
   "metadata": {},
   "outputs": [],
   "source": [
    "EPOCH = 100\n",
    "ALPHA = 0.1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "given-belgium",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy : 50.0%\n",
      "Accuracy : 75.0%\n",
      "Accuracy : 50.0%\n",
      "Accuracy : 75.0%\n",
      "Accuracy : 75.0%\n",
      "Accuracy : 100.0%\n",
      "Accuracy : 100.0%\n",
      "Accuracy : 100.0%\n",
      "Accuracy : 100.0%\n",
      "Accuracy : 100.0%\n"
     ]
    }
   ],
   "source": [
    "for i in range(1, EPOCH + 1):\n",
    "    randIndex = np.random.randint(0, len(input_data))\n",
    "    #Training\n",
    "    train_data = input_data[randIndex]\n",
    "    predict_output = feedForward(train_data)\n",
    "    \n",
    "    error = predict_output - target[randIndex]\n",
    "    \n",
    "    #update Weight\n",
    "    weight = weight - (np.array(train_data).reshape(2,1) * ALPHA * error)\n",
    "    \n",
    "    #updateBias\n",
    "    bias = bias - (ALPHA * error)\n",
    "    \n",
    "    if i % 10 == 0:\n",
    "        correct_result = 0\n",
    "        for i, test_data in enumerate(input_data):\n",
    "            test_result = feedForward(test_data)\n",
    "            if test_result == target[i]:\n",
    "                correct_result += 1\n",
    "        print(f'Accuracy : {correct_result / len(input_data) * 100}%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "destroyed-keyboard",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "driving-intellectual",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset = pd.read_csv('Iris.csv')\n",
    "#feature Selection/Extraction\n",
    "#Preprocessing\n",
    "\n",
    "input_data2 = dataset[['SepalLengthCm','SepalWidthCm']]\n",
    "target_data = dataset['Species']\n",
    "THESHOLD = 1\n",
    "\n",
    "target_data = np.where(target_data == 'Iris-setosa', 1, 0)\n",
    "target_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "toxic-cycling",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
