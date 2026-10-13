"""SDK protocol helpers and authenticated call context."""

from uuid import uuid4

from a2a.auth.user import User
from a2a.server.context import ServerCallContext
from a2a.types import a2a_pb2 as a2a
from google.protobuf.json_format import MessageToDict, ParseDict

TERMINAL = {a2a.TASK_STATE_COMPLETED, a2a.TASK_STATE_FAILED, a2a.TASK_STATE_CANCELED}
WAITING = TERMINAL | {a2a.TASK_STATE_INPUT_REQUIRED}


class Identity(User):
    def __init__(self, name):
        self.name = name

    @property
    def is_authenticated(self):
        return True

    @property
    def user_name(self):
        return self.name


def context(owner):
    return ServerCallContext(user=Identity(owner))


def to_dict(value):
    return MessageToDict(value)


def task_from(record):
    return ParseDict(record["snapshot"], a2a.Task())


def agent_message(task, text):
    return a2a.Message(
        message_id=str(uuid4()),
        task_id=task.id,
        context_id=task.context_id,
        role=a2a.ROLE_AGENT,
        parts=[a2a.Part(text=text)],
    )


def transition(record, state, text, reason="", **metadata):
    task = task_from(record)
    task.status.state = state
    task.status.timestamp.GetCurrentTime()
    task.status.message.CopyFrom(agent_message(task, text))
    task.metadata.update({"waiting_reason": reason, **metadata})
    if state in WAITING:
        task.history.append(task.status.message)
    record["snapshot"] = to_dict(task)
    record["reason"] = reason
    return task
