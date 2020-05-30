from flask import Flask, render_template, request, redirect
import code_logic
import io


from flask import Flask, make_response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)


@app.route('/')
def index_root():
    return render_template("Signin.html")


@app.route('/signin', methods=['POST', 'GET'])
def sign_in():
    if request.method == "POST":
        usrname = request.form["username"]
        pasword = request.form["password"]
        if(usrname== 'admin' and pasword == 'password'):
            return redirect('/view_graph')
        else:
            info = "Invalid username or password"
            return render_template("Signin.html", info=info)
    return render_template("Signin.html")


@app.route('/view_graph', methods=['POST', 'GET'])
def view_graph():
    curren = code_logic.get_currencies()
    if request.method == "POST":
        bar = request.form["bars"]
        curr = request.form["currency"]
        prd = request.form["period"]
        m_value = request.form["market_value"]
        code_logic.set_data(bars=bar, currency=curr, period=prd, market_value=m_value)
        codedata = code_logic.Data_Collect()
        msg = "yes"
        #return redirect('/view_graph')

        return render_template("datainfo.html",  currencies=curren, codedata=codedata, msg=msg)

    return render_template("datainfo.html", currencies=curren)



# @app.route('/display_graph', methods=['POST', 'GET'])
# def display_graph():
#     if request.method == "POST":
#         bar = request.form["bars"]
#         curr = request.form["currency"]
#         prd = request.form["period"]
#         m_value = request.form["market_value"]
#         code_logic.set_data(bars=bar, currency=curr, period=prd, market_value=m_value)
#         code_logic.Data_Collect()
#         #return redirect('/view_graph')
#         return render_template("name.html")
#     return redirect('/display_graph')


if __name__ == '__main__':
    app.run()
    # for run on specific ip address of server and port
    # app.run(host='127.0.0.1', port='5555')
