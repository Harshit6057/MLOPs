from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST']) # To render Homepage
def home_page():
    return render_template('index.html')


@app.route('/math', methods=['POST'])  # This will be called from UI
def math_operation():
    if (request.method == 'POST'):
        operation = request.form['operation']
        num1 = int(request.form['num1'])
        num2 = int(request.form['num2'])

        if(operation == 'add'):
            r = num1 + num2
            result = f"The sum of {str(num1)} and {str(num2)} is {str(r)}"

        if(operation == 'subtract'):
            r = num1 - num2
            result = f"The substract of {str(num1)} and {str(num2)} is {str(r)}"
        
        if(operation == 'multiply'):
            r = num1 * num2
            result = f"The multiply of {str(num1)} and {str(num2)} is {str(r)}"

        if(operation == 'divide'):
            r = num1 / num2
            result = f"The Division of {str(num1)} and {str(num2)} is {str(r)}"

        return render_template("results.html", result = result)

        
            


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)