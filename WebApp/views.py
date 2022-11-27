from django.shortcuts import render,HttpResponse

# Create your views here.
def appindex(request):
   
    return render(request, 'try.html')
def dia(request):
    return render(request, 'dia.html') 
def heart(request):
    return render(request, 'heart.html')       
def hresult(request):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    heart_data = pd.read_csv('heart.csv')
    heart_data.head()
    heart_data.tail()
    heart_data.shape
    heart_data.info()    
    heart_data.isnull().sum()
    heart_data.describe()
    heart_data['target'].value_counts()
    X = heart_data.drop(columns='target', axis=1)
    Y = heart_data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    val1=float(request.GET['hid1'])
    print(val1)
    val2=float(request.GET['hid2'])
    val3=float(request.GET['hid3'])
    val4=float(request.GET['hid4'])
    val5=float(request.GET['hid5'])
    val6=float(request.GET['hid6'])
    val7=float(request.GET['hid7'])
    val8=float(request.GET['hid8'])
    val9=float(request.GET['hid9'])
    val10=float(request.GET['hid10'])
    val11=float(request.GET['hid11'])
    val12=float(request.GET['hid12'])
    val13=float(request.GET['hid13'])

    input_data=(val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val11,val12,val13)
    input_data_as_numpy_array= np.asarray(input_data)

    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    hprediction = model.predict(input_data_reshaped)
    hresult=""
    print(hprediction)

    if (hprediction[0]== 0):
        hresult='The Person does not have a Heart Disease'
    else:
        hresult='The Person has Heart Disease'
    return render(request, 'heart.html',{"hresult":hresult})       



def result(request):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.svm import LinearSVC, SVC
    diabetic_dataset=pd.read_csv('diabetes.csv')
    diabetic_dataset.head()
    # no of row and column in dataset
    diabetic_dataset.shape
    diabetic_dataset.describe()
    diabetic_dataset['Outcome'].value_counts()
    diabetic_dataset.groupby('Outcome').mean()
    x=diabetic_dataset.drop(columns='Outcome',axis=1)
    y=diabetic_dataset['Outcome']
    scaler=StandardScaler()
    scaler.fit(x)
    standard_data=scaler.transform(x)
    x=standard_data
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,stratify=y,random_state=2)
    classifier=LinearSVC()
    classifier.fit(x_train, y_train)
    x_train_pridiction=classifier.predict(x_train)

    training_data_accuracy=accuracy_score(x_train_pridiction,y_train)
    x_test_pridiction=classifier.predict(x_test)
    test_data_accuracy=accuracy_score(x_test_pridiction,y_test)
    print(test_data_accuracy)
    print("%")
    val1=float(request.GET['id1'])
    print(val1)
    val2=float(request.GET['id2'])
    val3=float(request.GET['id3'])
    val4=float(request.GET['id4'])
    val5=float(request.GET['id5'])
    val6=float(request.GET['id6'])
    val7=float(request.GET['id7'])
    val8=float(request.GET['id8'])
    input_data=(val1,val2,val3,val4,val5,val6,val7,val8)
    print(input_data)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    fresult = scaler.transform(input_data_reshaped)
    prediction = classifier.predict(fresult)
    print(prediction[0])

    
    if (prediction[0] == 0):
        fresult="Not Diabetic"
    else:
       fresult="Diabetic"


    
    return render(request, 'dia.html',{"fresult":fresult})   
def trial(request):
    return render(request, 'try.html')     
   