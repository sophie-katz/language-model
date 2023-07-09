# Copyright (c) 2023 Sophie Katz
#
# This file is part of Language Model.
#
# Language Model is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# Language Model is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Language
# Model. If not, see <https://www.gnu.org/licenses/>.

from collections.abc import Iterable

import gradio as gr

from language_model.api import LanguageModelAPI


if __name__ == "__main__":
    api = LanguageModelAPI(50)

    print("Loading API...")
    api.load()

    def finish_sentence_accumulated(prompt_text: str) -> Iterable[str]:
        accumulator = ""

        for token in api.finish_sentence(prompt_text):
            if len(accumulator) > 0:
                accumulator += " "

            accumulator += token

            yield accumulator

    interface = gr.Interface(
        finish_sentence_accumulated,
        inputs=gr.Textbox(
            lines=4, placeholder="Text to be completed...", label="Prompt text"
        ),
        outputs=gr.Textbox(lines=32, label="Generated text"),
    )

    interface.queue().launch()
