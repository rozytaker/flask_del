import numpy as np
from flask import Flask, jsonify, make_response
from flask_restplus import Resource, Api, fields
from flask import request, stream_with_context, Response
from flask_cors import CORS
import json, csv
from werkzeug.utils import cached_property
import pickle
import pandas as pd
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
api = Api(app)

dummy_dm = api.model('dummy_dm', {
        'tenant_id': fields.String(description='Tenant Id', required=True, example='jaipur.com')
    })

model_new = keras.models.load_model('my_model')

# @app.route('/')
# def home():
#     return render_template('index.html')

@api.route('/prediction')
class LargeNew(Resource):
    @api.expect(dummy_dm)
    # def post(self):
    #     data = request.json
    #     tenant_id = data['tenant_id']
    #     with open('aggtrend_300_days.json') as f:
    #         output_dictionary = json.load(f)

    #     response = make_response(json.dumps(output_dictionary))
    #     response.headers['content-type'] = 'application/octet-stream'
    #     return response
    def post(self):
    
        import pandas_datareader as pdr

    #     df = pdr.get_data_tiingo('AAPL', api_key='37237cdbd35879937e842bda4e029d72c74892db')
    #     df.to_csv('AAPL.csv')
        import pandas as pd
        df=pd.read_csv('AAPL.csv')

        df1=df.reset_index()['close']
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler(feature_range=(0,1))
        df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

        training_size=int(len(df1)*0.65)
        test_size=len(df1)-training_size
        train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
        print('train_data',train_data.shape)
        print('test_data',test_data.shape)

        x_input=test_data[341:].reshape(1,-1)
        x_input.shape
        
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        print('tempinput',len(temp_input))
        # demonstrate prediction for next 10 days
        # demonstrate prediction for next 10 days
        from numpy import array

        lst_output=[]
        n_steps=100
        i=0
        while(i<30):

            if(len(temp_input)>100):
                #print(temp_input)
                x_input=np.array(temp_input[1:])
                print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
                yhat = model_new.predict(x_input, verbose=0)
                print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model_new.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1


        print(lst_output)
        day_new=np.arange(1,101)
        day_pred=np.arange(101,131)
        df3=df1.tolist()
        df3.extend(lst_output)

    #     df3=scaler.inverse_transform(df3).tolist()
        df3=scaler.inverse_transform(df3)
        pred=pd.DataFrame(df3)
        pred['Time']='2020-11-12'
        pred.columns=['pred','timestamp']
        print(pred.head(3))
        print('out',json.dumps(pred.to_dict(orient='records')))
        # pred.head(3)
        response = make_response(json.dumps(pred.to_dict(orient='records')))
        response.headers['content-type'] = 'application/octet-stream'
        return response
        

if __name__ == "__main__":
    app.run(debug=True)
