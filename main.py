# -*- coding: utf-8 -*-
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.compose import ColumnTransformer # FutureWarning: The handling of integer data will change in version 0.22

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder

from scipy.sparse import csr_matrix

import pandas as pd
import numpy as np

import sys, os, time


class PreProcessing:
    
    AuxDescriptive = None
    Descriptive = None 
    Target = None
    
    
    NColumns = 0
    flagDataset = 0
    flagPrecision = 0
    CategoricalColumns =  []
    
    def getCategorical(self, database):
        #database = pd.read_csv(file, header=None)
        # print("Dataset specifications \n")
        # print(database.info())
        # print("\n")
       
        numCols = len(database.columns) - 1
        self.CategoricalColumns=list(database.select_dtypes(include='object').columns)
        
        #retirar o indice da ultima coluna(target)
        try:
            self.CategoricalColumns.remove(numCols)
        except ValueError:
            print("Coluna " + str(numCols) + " não faz parte da lista...")
             
        
        self.NColumns = numCols
        
        
    
    #get column indexs search by val
    def getColumnsByVal(self, database, value):
        columnIndexes = list()
        result = database.isin([value])
        obj = result.any()
        columnIndexes = list(obj[obj == True ].index)
        return columnIndexes
    
    def cleanDatasetMostFrequent(self, database):
       
        imp = SimpleImputer(missing_values = np.nan , strategy= 'most_frequent')
        dsAux = imp.fit_transform(database)
        database = pd.DataFrame(dsAux, index = database.index, columns=database.columns)
        return database
        
    def cleanDatasetReplaceByZero(self, database):
        database = database.fillna('0')
        return database
    
    def preEncode(self,data):
        encoder = OrdinalEncoder()
        '''function to encode non-null data and replace it in the original data'''
        #retains only non-null values
        nonulls = np.array(data.dropna())
        #reshapes the data for encoding
        impute_reshape = nonulls.reshape(-1,1)
        #encode date
        impute_ordinal = encoder.fit_transform(impute_reshape)
        #Assign back encoded values to non-null values
        data.loc[data.notnull()] = np.squeeze(impute_ordinal)
        return data
    
    def cleanDatasetKNNBased(self, database):
       
        imputer = KNNImputer(n_neighbors=100) #n_neighbors=2
        database = database.drop([self.NColumns], axis=1)
        #database = imputer.fit_transform(database)
        for columns in self.CategoricalColumns:
           self.preEncode(database[columns])
        
        encode_data = pd.DataFrame(np.round(imputer.fit_transform(database)),columns = database.columns)
        print("pre_encode")
        return encode_data
        
    
    def readDataset(self, file):
        database = pd.read_csv(file, header=None, na_values=" ?")
        self.getCategorical(database)
        
        lymphographyColumnsNames = ['class','lymphatics','block_of_affere','bl_of_lymph_c','bl_of_lymph_s','by_pass','extravasates','regeneration_of','early_uptake_in','lym_nodes_dimin','lym_nodes_enlar','changes_in_lym','defect_in_node','changes_in_node','changes_in_stru','special_forms','dislocation_of','exclusion_of_no',' no_of_nodes']
        
        adultColumnsNames = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','result']
        
        if file =='.\\datasets\\lymphography.data':
            database.columns = lymphographyColumnsNames
            myTarget = 0
        else:
            database.columns = adultColumnsNames
            myTarget = 14
            
        # print(database)   
        #check if has null values(' ?' ==> nan )
        if database.isnull().values.any():
            print("Dataset com valores nulos, limpeza em curso...")
            aux_database=self.cleanDatasetMostFrequent(database)
            # aux_database=self.cleanDatasetReplaceByZero(database)
            # aux_database = self.cleanDatasetKNNBased(database)
        else:
            print("Dataset sem valores nulos!")
            aux_database = database
            
        # print(aux_database.head(50))
        print("Número de colunas: " + str(self.NColumns) + "\n")
        print("Lista Colunas Categóricas:" + str(self.CategoricalColumns))
        
        
        #para o dataset lymphography como target utilizamos a 1ª coluna
        if myTarget == 14:
            self.Descriptive = aux_database.iloc[:,0:self.NColumns].values
            self.Target = database.iloc[:,self.NColumns].values
            
            self.AuxDescriptive =  aux_database.iloc[:,0:self.NColumns].values
        else:
            self.Descriptive = aux_database.iloc[:,1:self.NColumns].values
            self.Target = database.iloc[:,0].values
            
            self.AuxDescriptive =  aux_database.iloc[:,1:self.NColumns].values
        
        
        
        self.flagDataset = 1
        # return database
        

        
    def labelEncoder(self):
        
        le = LabelEncoder()
        if len(self.CategoricalColumns):#apenas aplica o label encoder nas colunas do tipo categoricas
            print("Label encoding...\n")
            for i in self.CategoricalColumns:
            # for i in range(self.NColumns):
                self.Descriptive[:,i] = le.fit_transform(self.Descriptive[:,i])
        
        
    #duvida ---> deve-se aplicar a todas as colunas ? categóricas e continuas  -->self.CategoricalColumns
    def oneHotEncoder(self):
        #list(range(self.NColumns))
        he = ColumnTransformer([('one_hot_encoder',OneHotEncoder(), list(range(self.NColumns - 1)) )], remainder='passthrough') #kill warning
        # he = ColumnTransformer([('one_hot_encoder',OneHotEncoder(), self.CategoricalColumns )], remainder='passthrough')
        self.Descriptive = he.fit_transform(self.Descriptive)
         
 
        
    def standarScaler(self):
            ss = StandardScaler(with_mean=False) #ValueError: Cannot center sparse matrices: pass `with_mean=False` instead
            self.Descriptive = ss.fit_transform(self.Descriptive)
            

            
    def encoder(self, precision):
        self.flagPrecision = 1
        
        if precision == 'labelEncoder':
            self.labelEncoder()
        elif precision == 'oneHotEncoder':
            self.oneHotEncoder()
        elif precision == 'standardScaler':
            self.standarScaler()
        else:
            self.flagPrecision = 0


class Processing:
    
    SplitData = None
    Classifier = None
    
    Descriptive = None
    Target = None
    TestSize = 0
    Neighbors = 0
    def __init__(self, descriptive, target, testsize):
        self.Descriptive = descriptive
        self.Target = target
        self.TestSize = testsize
  
        
    def toDense(self,matrix):
        A = csr_matrix(matrix,dtype=float)
        Aux = A.todense()
        return Aux
        
    def naiveBayes(self):
        self.Classifier = GaussianNB()
            
        
    def decisionTree(self):
        self.Classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        
    def randomForest(self):
        self.Classifier = RandomForestClassifier(n_estimators=30, criterion='entropy', random_state=0)
        
    def kNN(self):
        numberNeighbors = int(self.Neighbors)
        self.Classifier = KNeighborsClassifier(n_neighbors=numberNeighbors, metric='minkowski', p=2, n_jobs = 1)
    
    def calculate(self, algorithm):
        start_time = time.time()
        if algorithm == 'naiveBayes':
            self.naiveBayes()
        if algorithm == 'decisionTree':
            self.decisionTree()
        if algorithm == 'randomForest':
            self.randomForest()
        if algorithm == 'kNN':
            self.kNN()
            
        # split data
        descriptiveTraining, descriptiveTest, targetTraining, targetTest = train_test_split(self.Descriptive, self.Target, test_size = self.TestSize, random_state=0)
        
        descriptiveTraining = self.toDense(descriptiveTraining)
        descriptiveTest = self.toDense(descriptiveTest)
        
        # descriptiveTraining = descriptiveTraining.toarray()
        #get prediction
        self.Classifier.fit(descriptiveTraining, targetTraining)
        
        prediction = self.Classifier.predict(descriptiveTest)
        accuracy = accuracy_score(targetTest,prediction)
        matrix = confusion_matrix(targetTest,prediction)
        
        print("Prediction:\n" + str(prediction) + "\n")
        print("Target:\n" + str(targetTest) +"\n")
        print("Accuracy: " + str(round(accuracy,2)) + "\n")
        print("Matrix:\n" + str(matrix) + "\n") 
        print("--- %s segundos ---" % (time.time() - start_time))
    
        
       
 
    


#############################TEST AREA#########################################
# PP = PreProcessing()
# PP.readDataset('.\\datasets\\adult.data')
# # # PP.getInfo('.\\datasets\\adult.data')
# # # PP.getInfo('.\\datasets\\lymphography.data')
# # a = PP.Descriptive
# PP.encoder('labelEncoder')
# # b = PP.Descriptive
# PP.encoder('oneHotEncoder')
# # c = PP.Descriptive
# #PP.encoder('standardScaler')
# d = PP.Descriptive

# P = Processing(PP.Descriptive, PP.Target, 0.25)
# P.calculate('naiveBayes')

###############################################################################
menu_actions  = {}
PP = PreProcessing()
def main_menu():
    os.system('clear')
    
    try:
      
        print ('\n\nSelcione uma opção:')
        print ('1. Escolher Dataset')
        print ('2. Escolher Precisão')
        print ('3. Escolher Algoritmo')
        print ('\n0. Sair')
        choice = input(' >>  ')
        exec_menu(choice)
    except SystemExit:
        print('Bye')
 
    return

         
def menu1():
    print ('Selecionar dataset')
    print ('1. Lymphography')
    print ('2. Adult')
    print ('9. Voltar')
    print ('0. Sair')
    choice = input(' >>  ')
    action_file(choice)
    return

def menu2():
    PP.Descriptive = PP.AuxDescriptive
   
    print ('Selecionar precisão')
    print ('1. labelEncoder')
    print ('2. labelEncoder + standardScaler')
    print ('3. labelEncoder + oneHotEncoder')
    print ('4. labelEncoder + oneHotEncoder + standardScaler')
    print ('9. Voltar(Dataset original)')
    print ('0. Sair' )
    choice = input(' >>  ')
    action_precision(choice)
    return

def menu3():
    print ('Selecionar algoritmo')
    print ('1. Naive Bayes')
    print ('2. Decision Tree')
    print ('3. Random Forest')
    print ('4. kNN')
    print ('9. Voltar')
    print ('0. Sair' )
    choice = input(' >>  ')
    action_algorithm(choice)
    return


def back():
    menu_actions['main_menu']()


def exitApp():
    sys.exit(1)
 
    
menu_actions = {
    'main_menu': main_menu,
    '1': menu1,
    '2': menu2,
    '3': menu3,
    '9': back,
    '0': exitApp
}

def exec_menu(choice):
    os.system('clear')
    ch = choice.lower()
    if ch == '':
        menu_actions['main_menu']()
    else:
        try:
            menu_actions[ch]()
        except KeyError:
            print ('Escolha Inválida!\n')
            menu_actions['main_menu']()
    return

def action_file(ch):
    global PP
   
    ch = ch.lower()
    os.system('clear')
    if ch == '1':
        File =  '.\\datasets\\lymphography.data'
        print('Dataset Lymphography selecionado\n')
    elif ch == '2':
        File = '.\\datasets\\adult.data'
        print('Dataset Adult selecionado\n')
    elif ch == '9':
        back()
    elif ch == '0':
        exitApp()
    else:
        print('Escolha inválida!!!\n')
        menu1()
        
    PP.readDataset(File)
    menu_actions['main_menu']()
    

def action_precision(ch):
    global PP
    ch = ch.lower()
    os.system('clear')
    
     
    if PP.flagDataset == 0:
        print('Deve selecionar um dataset primeiro!\n')
        menu1()
        
        
    if ch == '1':
        PP.encoder('labelEncoder')
    elif ch == '2':
        PP.encoder('labelEncoder')
        PP.encoder('standardScaler')
    elif ch == '3':
        PP.encoder('labelEncoder')
        PP.encoder('oneHotEncoder')
    elif ch == '4':
        PP.encoder('labelEncoder')
        PP.encoder('oneHotEncoder')
        PP.encoder('standardScaler')
    elif ch == '9':
        back()
    elif ch == '0':
        exitApp()
    else:
        print('Escolha inválida!!!\n')
        menu2()

    menu_actions['main_menu']()


def action_algorithm(ch):
    global PP
    
    
    ch = ch.lower()
    os.system('clear')
    algorithm=''
    
    if PP.flagDataset==0: 
        print('Deve selecionar um dataset primeiro!\n')
        menu1()
        
    if PP.flagPrecision==0:
        print('Deve selecionar a precisão antes!\n')
        menu2()

    neighbors=0
    
    if ch == '1':
        algorithm = 'naiveBayes'
    elif ch == '2':
        algorithm = 'decisionTree'
    elif ch == '3':
        algorithm = 'randomForest'
    elif ch == '4':
        neighbors = input("Nr vizinhos: ")
        algorithm = 'kNN'
    elif ch == '9':
        back()
    elif ch == '0':
        exitApp()
    else:
        print('Escolha inválida!!!\n')
        menu3()
  
    try:
        testSize = float(input('Insira o tamanho de testes(ex: 0.25): ' ))
        if testSize<=0 or testSize >1:
            print("Entre um número decimal entre 0 e 1")
            menu3()
    except ValueError:
        print("Entre um número decimal entre 0 e 1")
        menu3()
    
    P = Processing(PP.Descriptive, PP.Target, testSize)
    P.Neighbors = neighbors
    P.calculate(algorithm)
    
    menu_actions['main_menu']()
    
    
    


if __name__ == '__main__':  
    main_menu()
    

