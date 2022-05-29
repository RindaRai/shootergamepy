#создай здесь свой индивидуальный проект!
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix , accuracy_score

def f(value):
    i=0
    value = str(value)
    value_start = value
    while '.' in value:
        value = '0'+value
        if not(value.find('.')==1): 
            value = value[0:value.find('.')]+value[value.find('.')+1:-1]
            i+=1
    if i == 2:
        return str(value_start)
    else:
        return 'False'


def f2(value):
    year = int(value[len(value)-4:len(value)])
    if 2020 - year >= 18:
        return str(year)
    else:
        return 'False'



df = pd.read_csv('train.csv')

df['bdate'] = df['bdate'].apply(f)

df['city'] = df['city'].fillna('False')
df['occupation_type'] = df['occupation_type'].fillna('False')
df['education_form'] = df['education_form'].fillna('False')
df['bdate'] = df['bdate'].fillna('False')

df = df[df['city']!='False']
df = df[df['occupation_type']!='False']
df = df[df['education_form']!='False']
df = df[df['bdate']!='False']

df['bdate'] = df['bdate'].apply(f2)

df.drop('career_start', axis=1,inplace = True)
df.drop('career_end', axis=1,inplace = True)
df.drop('followers_count',axis=1,inplace = True)
df.drop('has_photo',axis=1,inplace = True)
df.drop('graduation',axis=1,inplace = True)
df.drop('last_seen',axis=1,inplace = True)
df.drop('occupation_name',axis=1,inplace = True)
df.drop('id',axis=1,inplace = True)
df.drop('has_mobile',axis=1,inplace = True)
df.drop('life_main', axis=1,inplace = True)
df.drop('people_main', axis=1,inplace = True)
df.drop('education_status', axis=1,inplace = True)

df['bdate'] =pd.get_dummies(df['bdate'])
df['education_form'] =pd.get_dummies(df['education_form'])
df['langs'] =pd.get_dummies(df['langs'])
df['city'] =pd.get_dummies(df['city'])
df['occupation_type'] =pd.get_dummies(df['occupation_type'])

x = df.drop('result', axis=1)
y = df['result']

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.25)



sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(xtrain,ytrain)

ypred = classifier.predict(xtest)
pr = accuracy_score(ytest,ypred)*100

i = 0
b = 0
for el in ypred:
    if el == 1:
        i += 1
    elif el == 0:
        b += 1

print(i/b*100)

print(pr)