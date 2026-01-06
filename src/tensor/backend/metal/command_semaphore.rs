use std::sync::{Condvar, Mutex, MutexGuard};

pub enum CommandStatus {
    Available,
    Encoding,
    Done,
}

pub struct CommandSemaphore {
    pub cond: Condvar,
    pub status: Mutex<CommandStatus>,
}

impl CommandSemaphore {
    pub fn new() -> Self {
        Self {
            cond: Condvar::new(),
            status: Mutex::new(CommandStatus::Available),
        }
    }

    pub fn wait_until<F: FnMut(&mut CommandStatus) -> bool>(
        &self,
        mut f: F,
    ) -> MutexGuard<'_, CommandStatus> {
        self.cond
            .wait_while(self.status.lock().unwrap(), |s| !f(s))
            .unwrap()
    }

    pub fn set_status(&self, status: CommandStatus) {
        *self.status.lock().unwrap() = status;
        self.cond.notify_one();
    }

    pub fn when<T, B: FnMut(&mut CommandStatus) -> bool, F: FnMut() -> T>(
        &self,
        b: B,
        mut f: F,
        next: Option<CommandStatus>,
    ) -> T {
        let mut guard = self.wait_until(b);
        let v = f();
        if let Some(status) = next {
            *guard = status;
            self.cond.notify_one();
        }
        v
    }
}
