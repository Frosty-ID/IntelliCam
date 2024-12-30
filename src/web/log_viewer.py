from flask import Flask, send_file

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <h1>Log Viewer</h1>
    <p><a href="/view-logs">View Logs</a></p>
    <p><a href="/download-logs">Download Logs</a></p>
    '''

@app.route('/view-logs')
def view_logs():
    try:
        with open('monitoring', 'r') as log_file:
            log_content = log_file.read()
        return f"<h1>Log Contents</h1><pre>{log_content}</pre>"
    except FileNotFoundError:
        return "<h1>Error:</h1><p>The log file does not exist.</p>"

@app.route('/download-logs')
def download_logs():
    try:
        return send_file('monitoring.log', as_attachment=True)
    except FileNotFoundError:
        return "<h1>Error:</h1><p>The log file does not exist.</p>"

if __name__ == '__main__':
    app.run(debug=True)
