#Database connection
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()


def create_page_visited_table():
	c.execute('create table if not exists pageTrackTable(pagename text,timeOfvisit timestamp)')

def add_page_visited_details(pagename,timeOfvisit):
	c.execute('insert into pageTrackTable(pagename,timeOfvisit) values(?,?)',(pagename,timeOfvisit))
	conn.commit()

def view_all_page_visited_details():
	c.execute('select * from pageTrackTable')
	data = c.fetchall()
	return data


# To Track Input & Prediction
def create_emotionclf_table():
	c.execute('create table if not exists emotionclfTable(rawtext text,prediction text,probability number,timeOfvisit timestamp)')

def add_prediction_details(rawtext,prediction,probability,timeOfvisit):
	c.execute('insert into emotionclfTable(rawtext,prediction,probability,timeOfvisit) values(?,?,?,?)',(rawtext,prediction,probability,timeOfvisit))
	conn.commit()

def view_all_prediction_details():
	c.execute('select * from emotionclfTable')
	data = c.fetchall()
	return data
