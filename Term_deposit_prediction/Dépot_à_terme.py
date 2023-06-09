
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler,FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib


# Journalisation

logging.basicConfig(
        filename="dépot_à_terme.log",
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')


# importation des données

def import_data(path):
    ''' 
    retourne le dataframe trouvé dans le chemin specifié 
    Input : Le chemin du dataframe

    output : dataframe
    
    '''
    df = pd.read_csv(path, sep=";")

    df.drop(["balance","duration"],axis = 1, inplace=True)

    df["y"] = df["y"].apply(lambda val: 0 if val == "no" else 1)

    return df

def data_spliting(df):

    '''
    Input : Dataframe
    
    Output : '''

    train, test = train_test_split(df, test_size=0.3,random_state= 123, stratify = df["y"])
    test, validate = train_test_split(test, test_size=0.5, random_state = 123, stratify = test["y"])

    train.to_csv('train.csv', index= False)
    test.to_csv('test.csv', index= False)
    validate.to_csv('validate.csv', index= False)

    X_train, X_val, Y_train, Y_val = train.drop('y', axis = 1), validate.drop('y', axis = 1), train['y'], validate['y']

    return train,X_train, X_val, Y_train, Y_val 

def perform_eda(df):

    df_copy= df.copy()
    columns = df_copy.columns.to_list()

    columns.append("heatmap")

    df_corr = df_copy.corr(numeric_only = True)

    for column in columns:
        plt.figure(figsize=(10,6))
        if column=='heatmap':
            sns.heatmap(
                df_corr,
                cmap='RdYlGn', annot=True, center=0)
            
        else:
            if df_copy[column].dtype !='O':
                df_copy[column].hist()
            else : 
                sns.countplot(data = df, x = column)
        plt.savefig('Projet1/images/EDA/' + column + '.jpg')
        plt.close()

    
def classifications_report(Y_train,
                           Y_preds_train,
                           Y_val,
                           Y_preds_val):
    class_reports_dico = {
        "logistic Regression train results": classification_report(Y_train, Y_preds_train),
        "logistic Regression validations results": classification_report(Y_val, Y_preds_val)
    }

    for title , report in class_reports_dico.items():
        plt.rc('figure', figsize = (7,3))

        plt.text(0.2, 0.3, str(report), {
            'fontsize':10}, fontproperties='monopaces')
        plt.axis('off')
        plt.title(title, fontweight="bold")
        plt.savefig('Projet1/images/ROC/' + title + '.jpg')
        plt.close()


# Fonction pour convertir la variable date

def convert_day(data):
    df = data.copy()
    df["day"]=df["day"].astype(object)
    return df.values

# Pipeline modèle

def build_pipeline():

    numeric_features = ['campaign', 
                        'pdays', 
                        'previous', 
                        'age']
    
    categorical_features = ['job',
                            'marital',
                            'education',
                            'default',
                            'housing',
                            'loan',
                            'contact',
                            'month',
                            'poutcome',
                            'day']
    
    # Pipeline de pretraitement des variables numeriques
    numeric_transformer = Pipeline(
        steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('Scaler', StandardScaler())
        ]
    )

    # Pipeline de pretraitement  des variables catégorielles
    categorical_transformer = Pipeline(
        steps =[
            ('convert', FunctionTransformer(convert_day)),
            ('OneHotEncoder', OneHotEncoder(handle_unknown='ignore'))]
    )

    preprocessor = ColumnTransformer(
        transformers=[('numeric',numeric_transformer, numeric_features),
                    ('categorical', categorical_transformer, categorical_features)]
    )

    # pipeline de modelisationx 
    pipeline_model = Pipeline(
        steps = [
            ('preprocessor', preprocessor),
            ('logreg', LogisticRegression(solver = 'lbfgs',
                                          random_state=123, 
                                          max_iter=2000,
                                          C = 0.5,
                                          penalty="l2"))
        ]
    )

    return  pipeline_model

# entrainement du modèle
def train_model(X_train, X_val, Y_train, Y_val):

    model = build_pipeline()

    model.fit(X_train, Y_train)

    y_preds_train  = model.predict(X_train)
    y_preds_val = model.predict(X_val)

    logreg_roc= RocCurveDisplay.from_estimator(model, X_val, Y_val)
    plt.savefig('images/ROC/roc_cuv.jpg')
    plt.close()


    classifications_report(Y_train,
                       y_preds_train,
                       Y_val,
                       y_preds_val)
    

    joblib.dump(model, 'model.pkl')

def main():
    logging.info("Importation des données...")
    raw_data = import_data("bank.csv")
    logging.info("Importation des données: SUCCES")

    logging.info("Division des données...")
    train_data, Xtrain, Ytrain, Xval, Yval = data_spliting(raw_data)
    logging.info("Division des données: SUCCES")

    logging.info("Analyse exploratoire des données...")
    perform_eda(train_data)
    logging.info("Analyse exploratoire des données: SUCCES")

    logging.info("Formation du modèle...")
    train_model(Xtrain, Ytrain, Xval, Yval)
    logging.info("Formation du modèle: SUCCES")

if __name__ == "__main__":
    print("execution en cours...")
    main()
    print("fin d el'execution avec succès")









