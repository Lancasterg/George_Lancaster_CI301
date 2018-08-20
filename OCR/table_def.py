#!/usr/bin/python
# -*- coding: utf-8 -*-
from sqlalchemy import *
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref, sessionmaker
from flask_login import LoginManager, UserMixin, login_required, \
    login_user, logout_user
from werkzeug.security import generate_password_hash, \
    check_password_hash
engine = create_engine('sqlite:///tutorial.db', echo=True)
Base = declarative_base()

''' This module defines the User and Prediciton objects for the database.

'''



class User(Base, UserMixin):

    '''User class
    '''

    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String)
    password = Column(String)

    def __init__(self, username, password):
        self.username = username
        self.set_password(password)

    def getId(userID):
        Session = sessionmaker(bind=engine)
        s = Session()
        query = s.query(User).filter(User.username == userID).all()
        user = User(query)
        print(user)
        return user

    def get_id(self):
        return self.username

    def set_password(self, password):
        self.password = generate_password_hash(password)


class Prediction(Base, UserMixin):

    '''Predicton class
    '''

    __tablename__ = 'prediction'
    id = Column(Integer, primary_key=True)
    user = Column('user_id', Integer, ForeignKey('users.id'),
                  nullable=False)
    image = Column(String)
    result = Column(String)

    def __init__(
        self,
        the_user,
        the_image,
        the_result,
        ):
        self.user = the_user
        self.image = the_image
        self.result = the_result


def result_query():
    Session = sessionmaker(bind=engine)
    s = Session()
    res = s.query(Prediction.id).filter(Prediction.user == '1')
    list_result = []
    for r in res:
        list_result.append(r[0])
    print(list_result)


def image_query():
    Session = sessionmaker(bind=engine)
    s = Session()
    res = s.query(User.id).filter(User.username == user.username)
    for r in res:
        identity = r.id


def query():
    Session = sessionmaker(bind=engine)
    s = Session()
    res = s.query(User.id).filter(User.username == 'admin')
    for r in res:
        identity = r.id
        print(identity)


def query_filename():
    Session = sessionmaker(bind=engine)
    s = Session()
    image_filename = s.query(Prediction.image).filter(Prediction.user
            == '1')
    for i in image_filename:
        print(i)


# query()
# query_filename()
# result_query()

# create tables
# Base.metadata.create_all(engine)
