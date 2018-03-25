import pandas
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# Task1
task = 1
f = open(f'titanic_answer_{task:d}.txt', 'w')

maleCount = data['Sex'].value_counts()['male']
femaleCount = data['Sex'].value_counts()['female']

answer = ' '.join(str(st) for st in (maleCount, femaleCount))
f.write(answer)
f.close()
# End of Task1

# Task2
task = 2
f = open(f'titanic_answer_{task:d}.txt', 'w')

count = data.count()['Survived']
alivePercent = round(data['Survived'].value_counts()[1] / count * 100, 2)

f.write(str(alivePercent))
f.close()
# End of Task3

# Task3
task = 3
f = open(f'titanic_answer_{task:d}.txt', 'w')

count = data.count()['Pclass']
firstClassPercent = round(data['Pclass'].value_counts()[1] / count * 100, 2)

f.write(str(firstClassPercent))
f.close()

# End of Task3

# Task4
task = 4
f = open(f'titanic_answer_{task:d}.txt', 'w')

meanAge = round(data['Age'].mean(), 2)
medianAge = data['Age'].median()

answer = ' '.join(str(st) for st in (meanAge, medianAge))

f.write(answer)
f.close()

# End of Task4

# Task5
task = 5
f = open(f'titanic_answer_{task:d}.txt', 'w')

corr = round(data['SibSp'].corr(data['Parch'], method='pearson'), 2)

f.write(str(corr))
f.close()

# End of Task5

# Task6
task = 6
f = open(f'titanic_answer_{task:d}.txt', 'w')

femaleData = data.loc[data['Sex'] == 'female']
femaleNames = femaleData['Name']
rightPart = femaleNames.str.split('.')
names = []

for right in rightPart:
    if '(' in right[1]:
        name = right[1].split('(')[1].split(' ')[0]
    else:
        name = right[1].split(' ')[1]
    names.append(name)

newData = pandas.DataFrame({'FirstName': names})
answer = newData['FirstName'].value_counts().idxmax()

f.write(answer)
f.close()

# End of Task6

