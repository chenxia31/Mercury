import pandas as pd
import graphviz
from sklearn import tree
df=pd.read_excel('originData.xlsx')

clf = tree.DecisionTreeClassifier()
clf = clf.fit(df[['S','K']], df['Correct or Not'])
tree.plot_tree(clf)


dot_data = tree.export_graphviz(clf, out_file=None,feature_names=df.columns[0:2],
                    class_names=df.columns[2],
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("jzh_correct")