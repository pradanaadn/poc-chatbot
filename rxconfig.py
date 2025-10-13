import reflex as rx

config = rx.Config(
    app_name="poc_chatbot",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)