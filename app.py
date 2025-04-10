from flask import Flask, render_template, flash, redirect, url_for, session, request, logging
from flask import Flask
from flask_mysqldb import MySQL
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps
from flask_login import login_user,LoginManager
import Ml_model as ml


import random
import nltk
nltk.download('stopwords')



app =  Flask(__name__)
app.secret_key = 'your-secret-key-here'

#generates a random string
def creuid():
    num1=0
    num2=0
    num1= random.randint(9999,99999)
    num2= random.randint(9999,99999)
    digits = len(str(num2))
    num1 = num1 * (10**digits)
    num1 += num2
    print(num1)
    return num1

num= random.randint(0, 1000)


app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'  # Default MySQL username
app.config['MYSQL_PASSWORD'] = 'Dhanu777%'   # Replace with your actual MySQL root password
app.config['MYSQL_DB'] = 'cybergaruna'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)



#adding Login Manager/login required class
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login')
            return redirect(url_for('login'))

    return wrap
#home
@app.route('/',methods=['GET', 'POST'])
def home():
    #get the count of predicted tweets
    count=ml.predt()
    return render_template('index.html', count=count)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        cur = None
        try:
            username = request.form['username']
            password_candidate = request.form['password']
            cur = mysql.connection.cursor()
            result = cur.execute("SELECT * FROM users WHERE username = %s", [username])
            if result > 0:
                data = cur.fetchone()
                password = data['password']
                if sha256_crypt.verify(password_candidate, password):
                    session['logged_in'] = True
                    session['username'] = username
                    return redirect(url_for('dashboard'))
                else:
                    flash("Check the Username and Password")
            else:
                flash('Username not found', 'error')
            return render_template('login.html')
        except Exception as e:
            flash('Error during login', 'error')
            return render_template('login.html')
        finally:
            if cur:
                cur.close()
    return render_template('login.html')



@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    #logout
    if request.method == 'POST':
        session.clear()
        flash("Successfully Logged Out")
        return redirect(url_for('login'))
 
 
 #dashboard
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    datam = []
    cur = None
    try:
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM regcases")   
        if result > 0:
            datam = cur.fetchall()
        return render_template('sampletable.html', data1=datam)
    except Exception as e:
        flash('Error loading dashboard', 'error')
        return render_template('sampletable.html', data1=datam)
    finally:
        if cur:
            cur.close()
        
@app.route('/accepted', methods=['GET', 'POST'])
@login_required
#table of accepted cases
def accepted():
    data = []
    cur = None
    try:
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM acceptedcomplaints")   
        if result > 0:
            data = cur.fetchall()
        return render_template('accepted_cases.html', data1=data)
    except Exception as e:
        flash('Error loading accepted cases', 'error')
        return render_template('accepted_cases.html', data1=data)
    finally:
        if cur:
            cur.close()

#table of rejected cases
@app.route('/rejected', methods=['GET', 'POST'])
@login_required
def rejected():
    data = []
    cur = None
    try:
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM rejectedcomplaints")   
        if result > 0:
            data = cur.fetchall()
        return render_template('rejected_cases.html', data1=data)
    except Exception as e:
        flash('Error loading rejected cases', 'error')
        return render_template('rejected_cases.html', data1=data)
    finally:
        if cur:
            cur.close()


#table of registered cases
@app.route('/register_complaint')
def regcomp():
    return render_template('regcomp.html')

#submit complaint
@app.route('/submit_complaint', methods=['GET', 'POST'])
def sub_comp():
    if request.method == 'POST':
        cur = None
        try:
            unid = creuid()
            case_type = request.form['ctype']
            case_desc = request.form['cdesc']
            case_proof = request.form['clink']
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO regcases(id, ctype, cdescription, clink) VALUES(%s, %s, %s, %s)", 
                       (unid, case_type, case_desc, case_proof))
            mysql.connection.commit()
            return render_template('reg_success.html', unid=str(unid))
        except Exception as e:
            if cur:
                mysql.connection.rollback()
            flash('Error submitting complaint', 'error')
            return render_template('regcomp.html')
    return render_template('regcomp.html')



@app.route('/check_complaint')

def chkcomp():
    return render_template('check.html')


#view complaints
@app.route('/view', methods=['GET', 'POST'])
@login_required
def getid():
    if request.method == 'POST':
        getval = request.form['getval']
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM regcases WHERE id = %s", [getval])
        if result > 0:
            data = cur.fetchone()
            case_id = data['id']
            case_type = data['ctype']
            case_desc = data['cdescription']
            case_proof = data['clink']    
            return render_template('viewcase.html', getval=case_id, case_type=case_type, case_desc=case_desc, case_proof=case_proof)
    return redirect(url_for('dashboard'))

@app.route('/test_complaint', methods=['GET', 'POST'])
@login_required
def test_complaint():
    if request.method == 'POST':
        cur = None
        try:
            complaint_id = request.form.get('complaint_id')
            if not complaint_id:
                flash('No complaint ID provided', 'error')
                return redirect(url_for('dashboard'))
                
            cur = mysql.connection.cursor()
            result = cur.execute("SELECT * FROM regcases WHERE id = %s", [complaint_id])
            if result > 0:
                complaint = cur.fetchone()
                complaint_text = complaint['cdescription']
                
                if not complaint_text or len(complaint_text.strip()) == 0:
                    flash('Error: The complaint text is empty', 'error')
                    return redirect(url_for('dashboard'))
                
                # Test the complaint text using ML model
                test_result = ml.test_complaint(complaint_text)
                if test_result:
                    return render_template('test_result.html', 
                        complaint=complaint,
                        result=test_result,
                        is_bullying=test_result['is_bullying'],
                        confidence=test_result['confidence'] * 100,
                        processed_text=test_result['processed_text']
                    )
                else:
                    flash('The text could not be analyzed. It may contain no valid words or only gibberish.', 'error')
                    return redirect(url_for('dashboard'))
            else:
                flash('Complaint not found', 'error')
                return redirect(url_for('dashboard'))
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print("Error in test_complaint route:", error_details)
            flash(f'Error processing complaint: {str(e)}', 'error')
            return redirect(url_for('dashboard'))
        finally:
            if cur:
                cur.close()
    return redirect(url_for('dashboard'))

#add to accepted table
@app.route('/comp_add_res', methods=['GET', 'POST'])
@login_required
def addcse():
    if request.method == 'POST':
        getval=request.form['compres']
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM regcases WHERE id = %s", [getval])
        if result > 0:
            data = cur.fetchone()
            case_id=data['id']
            case_type=data['ctype']
            case_desc=data['cdescription']
            case_proof=data['clink'] 
            officer_name=session['username']
            cur.execute("DELETE FROM regcases WHERE id = %s", [getval])
            cur.execute("INSERT INTO acceptedcomplaints(id, ctype, cdescription, clink, officer) VALUES(%s, %s, %s, %s, %s)", (case_id, case_type, case_desc, case_proof, officer_name))
            mysql.connection.commit()
            cur.close()
    cur = mysql.connection.cursor()
    flash('Case is accepted!', 'success')
    result = cur.execute("SELECT * FROM regcases")    
    if result > 0:   
        data = cur.fetchall()
        return render_template('sampletable.html',data1 =data)
    else:
        flash("No cases found", 'danger')
        return render_template('sampletable.html')


#add to rejected table
@app.route('/comp_del_res', methods=['GET', 'POST'])
@login_required
def dellcse():
    
    if request.method == 'POST':
        getval=request.form['compres']
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM regcases WHERE id = %s", [getval])
        if result > 0:
            data = cur.fetchone()
            case_id=data['id']
            case_type=data['ctype']
            case_desc=data['cdescription']
            case_proof=data['clink']
            officer_name=session['username'] 
            cur.execute("DELETE FROM regcases WHERE id = %s", [getval])
            cur.execute("INSERT INTO rejectedcomplaints(id, ctype, cdescription, clink, officer) VALUES(%s, %s, %s, %s, %s)", (case_id, case_type, case_desc, case_proof, officer_name))
            flash('Case is rejected!', 'success')
            mysql.connection.commit()
            cur.close()
        else:
            flash("Case not found", 'danger')
            return render_template('sampletable.html')
    cur = mysql.connection.cursor()
    result = cur.execute("SELECT * FROM regcases")   
    if result > 0:   
        data = cur.fetchall()
        return render_template('sampletable.html',data1 =data)
    else:
        flash("No cases found", 'danger')
        return render_template('sampletable.html')

          
#check status            
@app.route('/chkstatus', methods=['GET', 'POST'])
def chkstatus():
        if request.method == 'POST':
            getid=request.form['getid']
            cur = mysql.connection.cursor()
            result = cur.execute("SELECT * FROM regcases WHERE id = %s", [getid])
            if result > 0:
                flash('Your case is yet to be checked', 'success')
                return render_template('check.html')
            
            result = cur.execute("SELECT * FROM acceptedcomplaints WHERE id = %s", [getid])
            if result > 0:
                flash('Your case is under review', 'success')
                return render_template('check.html')
                
            result = cur.execute("SELECT * FROM rejectedcomplaints WHERE id = %s", [getid])
            if result > 0:
                flash('Your case is rejected', 'danger')
                return render_template('check.html')
        
    

if __name__=="__main__":
    
    app.secret_key='secret123'
    app.run(host='127.0.0.1', debug=True)
