from math import exp
import random

# TODO: Calculate logistic
def logistic(x):
      return 1.0 / (1 + exp(-x))

# TODO: Calculate dot product of two lists
def dot(X, Y):
    dotproduct=0
    for i,j in zip(X,Y):
        dotproduct += i*j
    return dotproduct

# TODO: Calculate prediction based on model
def predict(model, point):
  '''
  Returns 1D array of probabilities
  that the class label == 1
  '''

  if type(point) == dict:
      point = point['features']

  x = dot(model, point)
  return logistic(x)

# TODO: Calculate accuracy of predictions on data
def accuracy(data, predictions):
    total = len(predictions)
    correct = 0
    for i in range(0, len(predictions)):
        if ((predictions[i] <= 0.5 and data[i]['label'] == 0) or (predictions[i] > 0.5 and data[i]['label'] == 1)):
            correct += 1
    return (0.0 + correct) / total

# TODO: Update model using learning rate and L2 regularization
def update(model, point, delta, rate, lam):
    # the model = weight vector 
    prediction = predict(model, point['features']) #logistic(dot(model, point['features']))
    
    for i in range(len(model)):
        wi = model[i]
        gradient = ((-1 * lam) * wi) + (point['features'][i] * (point['label'] - prediction))

        model[i] = wi + (gradient*rate)

def initialize_model(k):
    return [random.gauss(0, 1) for x in range(k)]

# TODO: Train model using training data
def train(data, epochs, rate, lam):
    model = initialize_model(len(data[0]['features']))
    for e in range(epochs):
        for x in range(len(data)): 
            point = random.choice(data)
            update(model, point, 0, rate, lam)
            
    return model   
        
def extract_features(raw):
    data = []
    for r in raw:
        point = {}
        point["label"] = (r['income'] == '>50K')

        features = []
        features.append(1.)
        features.append(float(r['age'])/100)
        features.append(float(r['education_num'])/20)
        features.append(r['marital'] == 'Married-civ-spouse')
        #TODO: Add more feature extraction rules here!
        point['features'] = features
        data.append(point)
    return data

# TODO: Tune your parameters for final submission
#try to separte data using skilearn? 
def submission(data):
    return train(data, 4, .01, 0.001)
    
