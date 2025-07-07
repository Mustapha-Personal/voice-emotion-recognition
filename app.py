from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

from sqlalchemy import or_
import base64
import os
from datetime import datetime, timedelta
import base64
import uuid
from predict_emotion import predict_emotion

os.environ["PATH"] += os.pathsep + "C:\\ffmpeg\\bin"

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///voiceinsight.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ----------------------- Models -----------------------

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    business_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    recordings = db.relationship('Recording', backref='user', lazy=True)

class Recording(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_name = db.Column(db.String(150), nullable=False)
    customer_email = db.Column(db.String(150))
    file_path = db.Column(db.Text, nullable=False)
    emotion = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# ----------------------- Routes -----------------------

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        business_name = request.form['business_name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return redirect(url_for('register'))

        hashed_pw = generate_password_hash(password)
        user = User(business_name=business_name, email=email, password_hash=hashed_pw)
        db.session.add(user)
        db.session.commit()
        flash('Account created! You can log in now.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['business_name'] = user.business_name
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('landing'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = User.query.get(user_id)

    recent_recordings = Recording.query \
        .filter_by(user_id=user_id) \
        .order_by(Recording.created_at.desc()) \
        .limit(5).all()

    total_recordings = Recording.query.filter_by(user_id=user_id).count()

    positive_emotions = ['happy', 'excited', 'neutral']
    
    happy_customers = Recording.query.filter(
        Recording.user_id == user_id,
        Recording.emotion.in_(positive_emotions)
    ).count()

    # This week's recordings
    start_of_week = datetime.utcnow() - timedelta(days=datetime.utcnow().weekday())
    this_week = Recording.query.filter(
        Recording.user_id == user_id,
        Recording.created_at >= start_of_week
    ).count()

    total = total_recordings
    positive = Recording.query.filter(
        Recording.user_id == user_id,
        Recording.emotion.in_(positive_emotions)
    ).count()
    avg_sentiment = f"{int((positive / total) * 100)}%" if total > 0 else "0%"

    return render_template('dashboard.html',
                           recent_recordings=recent_recordings,
                           total_recordings=total_recordings,
                           happy_customers=happy_customers,
                           this_week=this_week,
                           avg_sentiment=avg_sentiment)
@app.route('/record', methods=['GET', 'POST'])
def record():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        customer_name = request.form['customer_name']
        customer_email = request.form.get('customer_email')
        audio_data = request.form['audio_data']

        if not customer_name or not audio_data:
            flash('Missing required fields.', 'error')
            return redirect(url_for('record'))

        if "," in audio_data:
            audio_data = audio_data.split(",")[1]

        try:
            decoded_audio = base64.b64decode(audio_data)
            filename = f"{uuid.uuid4().hex}.wav"
            filepath = os.path.join("static", "audio", filename)
            with open(filepath, "wb") as f:
                f.write(decoded_audio)
        except Exception as e:
            flash("Failed to save audio file.", "error")
            print("Error decoding audio:", e)
            return redirect(url_for('record'))
        

        emotion = predict_emotion(filepath)
        new_record = Recording(
            customer_name=customer_name,
            customer_email=customer_email,
            file_path=filepath,
            emotion=emotion,
            user_id=session['user_id']
        )
        db.session.add(new_record)
        db.session.commit()

        flash('Last feedback is: ' + emotion, 'success')
        return render_template('record.html', emotion=emotion)

    return render_template('record.html', emotion=None)


@app.route('/history')
def history():
    user_id = session['user_id']
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '').strip().lower()
    emotion = request.args.get('emotion', '').strip().lower()
    per_page = 10

    query = Recording.query.filter_by(user_id=user_id)

    if search:
        query = query.filter(
            or_(
                Recording.customer_name.ilike(f'%{search}%'),
                Recording.customer_email.ilike(f'%{search}%')
            )
        )
    if emotion:
        query = query.filter(Recording.emotion == emotion)

    pagination = query.order_by(Recording.created_at.desc()).paginate(page=page, per_page=per_page)

    recordings = pagination.items

    return render_template(
        'history.html',
        recordings=recordings,
        pagination=pagination,
        total_recordings=query.count(),
        happy_count=query.filter(Recording.emotion == 'happy').count(),
        angry_count=query.filter(Recording.emotion == 'angry').count(),
        sad_count=query.filter(Recording.emotion == 'sad').count(),
        neutral_count=query.filter(Recording.emotion == 'neutral').count(),
        search=search,
        emotion_filter=emotion,
    )
# ----------------------- CLI -----------------------

@app.cli.command('init-db')
def init_db():
    db.create_all()
    print("✅ Database initialized.")

# ----------------------- Run -----------------------

if __name__ == '__main__':

    with app.app_context():
        db.create_all()
        print("✅ Database tables created.")
    app.run(debug=True)
