import pyglet

window = pyglet.window.Window(width=200, height=200, caption='Test')
label = pyglet.text.Label('Hello, world', x=window.width//2, y=window.height//2, anchor_x='center', anchor_y='center')

@window.event
def on_draw():
    window.clear()
    label.draw()

@window.event
def on_key_press(symbol, modifiers):
    print(f"Key pressed: {symbol}")
    if symbol == pyglet.window.key.Q:
        pyglet.app.exit()

pyglet.app.run()