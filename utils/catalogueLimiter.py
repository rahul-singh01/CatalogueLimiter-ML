import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import os, json

class CatalogScorer:
    def __init__(self, catalog, training_data=None):
        self.catalog = catalog
        self.weights = {
            'compliance': {'promotion': 0.3, 'total_weight': 0},
            'correctness': {'rating': 0.4, 'total_weight': 0},
            'completeness': {'attributes': 0.5, 'total_weight': 0}
        }
        self.model = RandomForestClassifier()  # Initialize the machine learning model
        if training_data:
            self.fit_model(training_data)

    def define_weights(self):
        self.weights['compliance']['total_weight'] = sum(self.weights['compliance'].values())
        self.weights['correctness']['total_weight'] = sum(self.weights['correctness'].values())
        self.weights['completeness']['total_weight'] = sum(self.weights['completeness'].values())

    def fit_model(self, training_data):
        X_train = [self.extract_features(example['product']) for example in training_data]
        y_train = [example['label'] for example in training_data]
        self.model.fit(X_train, y_train)

    def extract_features(self, product):
        return [product['price'], product['rating'], product['promotion']]

    def evaluate_compliance_ml(self):
        if not hasattr(self.model, 'predict'):
            raise ValueError("The model must be fitted before making predictions.")
        X = [self.extract_features(product) for product in self.catalog.get('products', [])]
        predictions = self.model.predict(X)
        normalized_compliance_score = sum(predictions) / len(predictions)
        return normalized_compliance_score

    def evaluate_correctness(self):
        total_correctness_score = 0
        for product in self.catalog.get('products', []):
            correctness_score = product.get('rating', 0) / 5.0  # Normalize rating between 0 and 1
            total_correctness_score += correctness_score
        normalized_correctness_score = total_correctness_score / len(self.catalog.get('products', []))
        return normalized_correctness_score

    def evaluate_completeness(self):
        total_completeness_score = 0
        for product in self.catalog.get('products', []):
            attributes = product.get('attributes', [])
            completeness_score = sum(1 for attr in attributes if product.get(attr) is not None) / len(attributes)
            total_completeness_score += completeness_score
        normalized_total_completeness_score = total_completeness_score / len(self.catalog.get('products', []))
        return normalized_total_completeness_score

    def compute_catalog_score(self):
        compliance_score_ml = self.evaluate_compliance_ml()
        correctness_score = self.evaluate_correctness()
        completeness_score = self.evaluate_completeness()

        total_score = (
            compliance_score_ml * self.weights['compliance']['total_weight'] +
            correctness_score * self.weights['correctness']['total_weight'] +
            completeness_score * self.weights['completeness']['total_weight']
        )

        return total_score

def catalogueRater(productList):
    productList = json.loads(json.loads(productList))
    
    df = pd.read_csv('ecomm.csv')

    df = df.fillna(0) 

    training_data = []
    for _, row in df.iterrows():
        label = 1 if row['Promotion'] == 'Yes' else 0
        product = {
            'price': row['Price'],
            'rating': row['Rating'],
            'promotion': 1 if row['Promotion'] == 'Yes' else 0,
            'attributes': ['ProductName', 'Category'],  # Add relevant attributes
        }
        training_data.append({'product': product, 'label': label})
        
    imputer = SimpleImputer(strategy='mean')  
    X_train = [[example['product']['price'], example['product']['rating'], example['product']['promotion']] for example in training_data]
    X_train_imputed = imputer.fit_transform(X_train)

    for i, example in enumerate(training_data):
        example['product']['price'] = X_train_imputed[i][0]
        example['product']['rating'] = X_train_imputed[i][1]
        example['product']['promotion'] = X_train_imputed[i][2]

    # the rating is biased on promotion prompt which is true of false (0 or 1)
    # 0 means no promotion 1 means promotion
    #also the rating is biased on rating value
    catalog_data = {
        'products': productList
        # [
            # {'price': 576.0, 'rating': 8.0, 'promotion': 1, 'attributes': ['Badminton Set', 'Electronics']},
            # {'price': 576.0, 'rating': 4.0, 'promotion': 1, 'attributes': ['Badminton Set', 'Electronics']},
            # {'price': 576.0, 'rating': 3.0, 'promotion': 1, 'attributes': ['Badminton Set', 'Electronics']},
            # {'price': 576.0, 'rating': 9.0, 'promotion': 1, 'attributes': ['Badminton Set', 'Electronics']},
            # {'price': 576.0, 'rating': 3.0, 'promotion': 0, 'attributes': ['Badminton Set', 'Electronics']},
            # {'price': 576.0, 'rating': 3.0, 'promotion': 0, 'attributes': ['Badminton Set', 'Electronics']},
            # Add more products for prediction
        # ]   
    }
    

    catalog_scorer = CatalogScorer(catalog_data, training_data)
    catalog_scorer.define_weights()

    catalog_score_ml = catalog_scorer.compute_catalog_score()
    
    return {
        "score" : catalog_score_ml,
        "data" : productList
    }
# print(catalogueRater(
#     [
#         {'price': 576.0, 'rating': 5.0, 'promotion': 0, 'attributes': ['Badminton Set', 'Electronics']}
#     ]
#     ))