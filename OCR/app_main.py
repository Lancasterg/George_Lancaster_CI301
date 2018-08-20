from flask import Flask, flash, redirect, render_template, request, session, abort
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user
from scipy.misc import imsave, imread, imresize
from werkzeug import secure_filename
from persistence import *
from back_end import *
import codecs
import cv2
from sqlalchemy.orm import sessionmaker
from table_def import *
from sqlalchemy.inspection import inspect
from werkzeug.security import generate_password_hash, \
     check_password_hash

''' This handles the setup of the web application. All client requests are
routed through this module to
'''

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
login_manager = LoginManager()
engine = create_engine('sqlite:///tutorial.db', echo=True)
Session = sessionmaker(bind=engine)
s = Session()
app = Flask(__name__)
login_manager.init_app(app)

#LoginManager
@login_manager.user_loader
def load_user(user_id):
    '''Load a user from the database using a user_id.

    Args:
        user_id: The id of the user to be found.

    Returns:
        res: The user where user.user_id == user_id.
    '''
    res = s.query(User).filter(User.username == user_id).first()
    return res

@app.route('/')
def home():
    ''' The home naviagion option.

    If a user is not logged in, show Login screen.
    else show the Hub.
    '''
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return hub()

# Login / Signup ------------------------------------------------------------
@app.route('/login', methods=['POST'])
def do_admin_login():
    '''Log a user into the application.
    if username and password match, log into the application.

    else call home().
    '''
    POST_USERNAME = str(request.form['username'])
    POST_PASSWORD = str(request.form['password'])
    result = s.query(User).filter(User.username.in_([POST_USERNAME])).first()
    if result:
        if check_password_hash(result.password, POST_PASSWORD):
            session['logged_in'] = True
            session['user'] = POST_USERNAME
            user = load_user(POST_USERNAME)
            login_user(user)
    else:
        return (render_template('login.html', password=False))
    return home()

@app.route('/logout')
@login_required
def logout():
    '''Log out of the application.

    Logs a user out of the application and shows the login screen.
    '''
    session['logged_in'] = False
    session['user'] = ''
    logout_user()
    return home()

@app.route('/signup')
def signup():
    ''' Load the signup page.

    '''
    return render_template('signup.html')

@app.route('/guest')
def guest():
    ''' Log in as a guest.

    Sets the current user to guest and loads the classify_anonymous page.
    '''
    session['logged_in'] = False
    session['user'] = 'guest'
    return render_template('classify_anonymous.html')

@app.route('/create_user', methods = ['POST'])
def create_user():
    ''' Create a new user.

    Takes information from the forms and creates a new entry in the database
    for the user, if the username does not already exist.
    '''
    POST_USERNAME = str(request.form['username'])
    POST_PASSWORD = str(request.form['password'])
    POST_PASSWORD_CONFIRM = str(request.form['confirm-password'])
    res = s.query(User).filter(User.username.in_([POST_USERNAME])).first()
    if res == None:
        if POST_PASSWORD == POST_PASSWORD_CONFIRM:
            user = User(POST_USERNAME,POST_PASSWORD)
            s.add(user)
            s.commit()
            session['logged_in'] = True
            session['user'] = POST_USERNAME
            user = load_user(POST_USERNAME)
            if not os.path.exists('static/users/{}'.format(user.username)):
                os.makedirs('static/users/{}'.format(user.username))
            login_user(user)
            return render_template('hub.html')
        else:
            return (render_template('signup.html', password=False))
    else:
        return (render_template('signup.html', username=False))
    return 'ok'

# Hub page ------------------------------------------------------------------
@app.route('/hub')
@login_required
def hub():
    '''Retrieve a users predictions and load the Hub page.
    '''
    zipped = get_images()
    return render_template('hub.html', zipped=zipped)

def get_images():
    '''Load the currently logged in users prediction history.
    This is used in the web application to populate the Hub page.

    Returns:
        A list of predictions for the current user.
    '''
    user = load_user(session['user'])
    image_ref = s.query(Prediction.id).filter(Prediction.user == user.id)
    list_result = []
    for ref in image_ref:
        list_result.append(ref[0])
    all_images = []
    image_filename = s.query(Prediction.image).filter(Prediction.user == user.id)
    for img in image_filename:
        all_images.append(img[0])
    zipped = [list(a) for a in zip(list_result, all_images)]
    print(zipped)
    zipped = reversed(zipped)
    return zipped

@app.route('/download_result/', methods=['GET','POST'])
def download_result():
    '''Create a text file using the data from a prediction in the database.

    Returns:
        The prediction.
    '''
    prediction_id = request.get_data()
    prediction_id = prediction_id.decode("utf-8")
    result = s.query(Prediction.result).filter(Prediction.id == prediction_id)
    for res in result:
        print(res)
    loc = 'static/users/{}/result.txt'.format(session['user'])
    return res[0]

@app.route('/delete_result/', methods=['POST'])
def delete_result():
    ''' Delete a prediciton from the database.
    '''
    prediction_id = request.get_data()
    prediction_id = prediction_id.decode("utf-8")
    s.query(Prediction).filter(Prediction.id == prediction_id).delete()
    s.commit()
    return(home())

# Prediction Page ------------------------------------------------------------
@app.route('/classify')
def index():
    '''Loads the classify page
    '''
    return render_template('classify.html')

# Decoding an image from base64 into raw representation
def convertImage(imgData1, path):
    '''Decodes an image from base64.

    Args:
        imgData1: The image to be decoded
        path: The location the decoded image is saved
    '''
    img_str = re.search(b'base64,(.*)', imgData1).group(1)
    with open(path + '/temp.png', 'wb') as output:
        output.write(codecs.decode(img_str, 'base64'))

# Predict the output
@app.route('/predict/', methods=['GET','POST'])
def predict():
    ''' Makes a prediciton from an image send from the application.

    The method retrieves the image, converts it to a usable format, makes a
    prediciton, then saves it if the user is not a guest.

    Returns:
        The result of the prediciton.
    '''
    img_data = request.get_data()
    path = 'static/users/{}'.format(session['user'])
    if not os.path.exists(path):
        os.makedirs(path)
    img_data = request.get_data()
    convertImage(img_data, path)
    result = str(get_result(path + '/temp.png'))
    if session['user'] != 'guest':
        save_pred(path + '/temp.png', result)

    return result

def save_pred(path, result):
    '''Saves the prediction to the users file.

    Args:
        path: the loaction to save the image
        result: the result of the prediction
    '''
    path = 'static/users/{}'.format(session['user'])
    save_loc = save_thumbnail(session['user'], cv2.imread('static/users/{}/temp.png'.format(session['user'])))
    os.remove('static/users/{}/temp.png'.format(session['user']))
    create_prediction(save_loc, result)

def create_prediction(loc, result):
    '''creates the prediction entry in the database.

    Args:
        loc: the location of the image
        result: the prediciton result
    '''
    user = load_user(session['user'])
    res = s.query(User.id).filter(User.username == user.username)
    for r in res:
        identity = r.id
    pred = Prediction(identity, loc, result)
    s.add(pred)
    s.commit()

# Set the type of network being used
@app.route('/setNetwork/',methods=['GET','POST'])
def setNetwork():
    '''Set the network to make the prediciton.

    Returns:
        The type of network to be used.
    '''
    data = request.get_data()
    set_network(str(data))
    return(data)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    '''Upload a file for prediction

    Returns:
        Confirmation string
    '''
    if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'

@app.route('/save_result/', methods=['GET','POST'])
def save_result():
    '''Save the image to the users folder.
    '''
    img_data = request.get_data()
    user = session['user']
    path = 'static/users/{}'.format(user)
    save_image(user, image_data, path)


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run( host='0.0.0.0', port=5000)
