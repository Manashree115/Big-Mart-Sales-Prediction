from flask import Flask, jsonify, render_template, request
from flask_cors import cross_origin
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/home')
def home():
    # redirect to the newpage.html file
    return render_template('home.html')  


@app.route("/")
@cross_origin()
def index():
    return render_template('home.html')
    

 

@app.route('/pred')
def pred():
    # redirect to the newpage.html file
    return render_template('pred.html')       

@app.route('/predict',methods=['POST','GET'])
def result():

    item_weight= float(request.form['item_weight'])
    item_fat_content=float(request.form['item_fat_content'])
    item_visibility= float(request.form['item_visibility'])
    item_type= float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year= float(request.form['outlet_establishment_year'])
    outlet_size= float(request.form['outlet_size'])
    outlet_location_type= float(request.form['outlet_location_type'])
    outlet_type= float(request.form['outlet_type'])

    X= np.array([[ item_weight,item_fat_content,item_visibility,item_type,item_mrp,
                  outlet_establishment_year,outlet_size,outlet_location_type,outlet_type ]])

    scaler_path= r'C:\Users\Aryan\Documents\bigmart_final\models\sc.sav'

    sc=joblib.load(scaler_path)

    X_std= sc.transform(X)

    model_path=r'C:\Users\Aryan\Documents\bigmart_final\models\lr.sav'

    model= joblib.load(model_path)

    Y_pred=model.predict(X_std)
    output=round(Y_pred[0],2)

    # Chart

    import pandas as pd
    import matplotlib.pyplot as plt

# Read CSV file into a pandas dataframe
    df = pd.read_csv('Test.csv')

# Get frequency count of each unique item in the "Item_Types" column
    item_type_counts = df['Item_Type'].value_counts()

# Plot a bar chart of the frequency counts
    plt.bar(item_type_counts.index, item_type_counts.values)
    plt.title('Item Type Frequency')
    plt.xlabel('Item Types')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90) # Rotate x-axis labels for readability
    plt.show()


    

    return render_template('pred.html',prediction_text="The predicted price of the item that you entered is Rs. {}".format(output))


    return render_template("home.html")

    

if __name__ == "__main__":
    app.run(debug=True)





# Load the CSV data into a pandas dataframe
