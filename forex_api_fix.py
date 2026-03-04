# Add these routes to your app.py
# Paste them anywhere after your existing routes

@app.route('/api/forex-analyze', methods=['POST'])
@login_required
def forex_analyze():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({'content': message.content[0].text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forex-picks', methods=['POST'])
@login_required
def forex_picks():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2200,
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({'content': message.content[0].text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
