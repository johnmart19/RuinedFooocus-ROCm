"""Serialize prompt assistance with chat and image model operations."""

import threading


def run_prompt(prompt, system_file=None, *, instruction=None, check_cancelled=None):
    from ruinedfooocus.app import async_worker as worker

    if threading.current_thread() is not worker.worker_thread:
        from ruinedfooocus.app.api_runtime import submit, wait_for_result

        def operation(job, task_id):
            def check():
                if check_cancelled:
                    check_cancelled()
                worker.check_interrupt({"_api_job": job})

            return run_prompt(prompt, system_file, instruction=instruction, check_cancelled=check)

        return wait_for_result(submit(operation))
    from ruinedfooocus.features.chat.llama_pipeline import run_llama

    return run_llama(system_file, prompt, instruction=instruction, check_cancelled=check_cancelled)
