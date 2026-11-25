//! The checks before a recording, without any screen: run them on a thread,
//! take each update, then leave with the outcome, offer "Start anyway", or wait
//! for Retry (the logic of Python's `PreflightScreen`).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use tracing::{error, warn};

use crate::app::preflight::{
    Check, PreflightOutcome, PreflightPlan, PreflightReport, Reload, new_checks, run_preflight,
};
use crate::app::tasks::{Payload, Tasks};
use crate::ports::transcriber::Transcriber;

/// One check changed, from the check thread.
pub struct CheckUpdate(pub Check);

/// The checks ended, from the check thread.
pub struct ChecksDone {
    pub result: Result<PreflightReport, String>,
    pub transcriber: Box<dyn Transcriber>,
}

/// The line under the checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Summary {
    None,
    /// Something blocks the start.
    Red(String),
    /// The start is possible, with a cost.
    Yellow(String),
}

pub struct PreflightRuntime {
    pub plan: Arc<PreflightPlan>,
    reload: Option<Reload>,
    transcriber: Option<Box<dyn Transcriber>>,
    pub checks: Vec<Check>,
    report: Option<PreflightReport>,
    pub running: bool,
    left: bool,
    pub summary: Summary,
    /// "Start anyway" shows: every failure can be overridden.
    pub show_override: bool,
}

impl PreflightRuntime {
    pub fn new(
        plan: PreflightPlan,
        transcriber: Box<dyn Transcriber>,
        reload: Option<Reload>,
    ) -> Self {
        Self {
            plan: Arc::new(plan),
            reload,
            transcriber: Some(transcriber),
            checks: new_checks(),
            report: None,
            running: false,
            left: false,
            summary: Summary::None,
            show_override: false,
        }
    }

    /// Run every check on a thread. Each update and the end come back as task
    /// results for `task_result`.
    pub fn begin(&mut self, tasks: &Tasks) {
        let Some(mut transcriber) = self.transcriber.take() else {
            return;
        };
        self.running = true;
        self.report = None;
        self.summary = Summary::None;
        self.show_override = false;
        let plan = self.plan.clone();
        let sender = tasks.clone();
        let spawned = std::thread::Builder::new()
            .name("ltn-preflight".into())
            .spawn(move || {
                let notify = |check: &Check| sender.post(CheckUpdate(check.clone()));
                // run_preflight catches per check; this is the last guard.
                let result = crate::app::tasks::catching(|| {
                    run_preflight(&plan, transcriber.as_mut(), &notify)
                })
                .inspect_err(|message| {
                    error!(target: "ltn.preflight", "preflight crashed: {message}");
                });
                if plan.cancel.load(Ordering::SeqCst) {
                    // Back was pressed while the checks ran; nobody uses this child.
                    transcriber.close();
                    return;
                }
                sender.post(ChecksDone {
                    result,
                    transcriber,
                });
            });
        if let Err(e) = spawned {
            error!(target: "ltn.preflight", "Cannot start the checks: {e}");
            self.running = false;
            self.summary = Summary::Red(format!("The checks stopped: {e}. Press Retry."));
        }
    }

    /// Read config.yaml again, then run the checks again.
    pub fn retry(&mut self, tasks: &Tasks) {
        if self.running {
            return;
        }
        if let Some(reload) = &self.reload {
            match reload() {
                Ok(plan) => self.plan = Arc::new(plan),
                Err(e) => {
                    warn!(target: "ltn.preflight", "Could not read the config again: {e}");
                    self.summary = Summary::Red(format!("Could not read config.yaml: {e}"));
                    return;
                }
            }
        }
        self.begin(tasks);
    }

    /// Leave without a session: stop the checks and close the whisper child.
    /// False when the screen already left.
    pub fn back(&mut self) -> bool {
        if self.left {
            return false;
        }
        self.left = true;
        self.plan.cancel.store(true, Ordering::SeqCst);
        if let Some(mut transcriber) = self.transcriber.take() {
            transcriber.close();
        }
        true
    }

    fn leave(&mut self, outcome: PreflightOutcome) -> Option<PreflightOutcome> {
        if self.left {
            return None;
        }
        self.left = true;
        Some(outcome)
    }

    /// Start with the failures the report allows. None while nothing allows it.
    pub fn start_anyway(&mut self) -> Option<PreflightOutcome> {
        let mut report = self.report.take()?;
        if report.blocked() {
            self.report = Some(report);
            return None;
        }
        let Some(transcriber) = self.transcriber.take() else {
            self.report = Some(report);
            return None;
        };
        report.overridden = true;
        self.leave(PreflightOutcome {
            report,
            plan: self.plan.clone(),
            transcriber,
        })
    }

    fn show_report(
        &mut self,
        report: PreflightReport,
        transcriber: Box<dyn Transcriber>,
    ) -> Option<PreflightOutcome> {
        self.running = false;
        self.checks = report.checks.clone();
        if report.ok() {
            return self.leave(PreflightOutcome {
                report,
                plan: self.plan.clone(),
                transcriber,
            });
        }
        self.transcriber = Some(transcriber);
        if report.blocked() {
            self.summary = Summary::Red("Fix the items marked ✗, then press Retry.".into());
        } else {
            self.summary = Summary::Yellow(
                match report.llm_ok() {
                    false => "You can start without AI notes: the transcript is still recorded.",
                    true => "You can start anyway, but this problem stays for the whole meeting.",
                }
                .into(),
            );
            self.show_override = true;
        }
        self.report = Some(report);
        None
    }

    /// A task result. `Ok(Some(outcome))`: the session may start. Gives the
    /// payload back when it is not a check result, or after the screen left.
    pub fn task_result(&mut self, payload: Payload) -> Result<Option<PreflightOutcome>, Payload> {
        if self.left {
            return Err(payload);
        }
        let payload = match payload.downcast::<CheckUpdate>() {
            Ok(update) => {
                let update = update.0;
                if let Some(check) = self.checks.iter_mut().find(|check| check.key == update.key) {
                    *check = update;
                }
                return Ok(None);
            }
            Err(payload) => payload,
        };
        let done = payload.downcast::<ChecksDone>()?;
        let ChecksDone {
            result,
            transcriber,
        } = *done;
        Ok(match result {
            Ok(report) => self.show_report(report, transcriber),
            Err(message) => {
                self.running = false;
                self.transcriber = Some(transcriber);
                self.summary = Summary::Red(format!("The checks stopped: {message}. Press Retry."));
                None
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::preflight::Status;
    use crate::app::preflight::tests::{Provider, Whisper, plan, runtime};

    fn report(failed: &[&str]) -> PreflightReport {
        let mut checks = new_checks();
        for check in &mut checks {
            match failed.contains(&check.key) {
                true => check.set(Status::Fail, "broken", "fix it"),
                false => check.set(Status::Pass, "", ""),
            }
        }
        PreflightReport {
            checks,
            overridden: false,
        }
    }

    fn done(failed: &[&str]) -> Payload {
        Box::new(ChecksDone {
            result: Ok(report(failed)),
            transcriber: Box::new(Whisper::default()),
        })
    }

    #[test]
    fn a_clean_report_hands_back_the_outcome() {
        let dir = tempfile::tempdir().unwrap();
        let tokio = runtime();
        let plan = plan(dir.path(), "{}", Provider::Answers, &tokio);
        let mut preflight = PreflightRuntime::new(plan, Box::new(Whisper::default()), None);
        let outcome = preflight.task_result(done(&[])).ok().flatten();
        assert!(outcome.is_some_and(|outcome| outcome.report.ok()));
    }

    #[test]
    fn a_soft_failure_waits_for_start_anyway() {
        let dir = tempfile::tempdir().unwrap();
        let tokio = runtime();
        let plan = plan(dir.path(), "{}", Provider::Answers, &tokio);
        let mut preflight = PreflightRuntime::new(plan, Box::new(Whisper::default()), None);
        assert!(matches!(preflight.task_result(done(&["llm"])), Ok(None)));
        assert!(preflight.show_override);
        let outcome = preflight.start_anyway().expect("start anyway");
        assert!(outcome.report.overridden);
        assert!(preflight.start_anyway().is_none(), "it leaves once");
    }

    #[test]
    fn after_back_the_results_are_given_back() {
        let dir = tempfile::tempdir().unwrap();
        let tokio = runtime();
        let plan = plan(dir.path(), "{}", Provider::Answers, &tokio);
        let mut preflight = PreflightRuntime::new(plan, Box::new(Whisper::default()), None);
        assert!(preflight.back());
        assert!(!preflight.back(), "a second back does nothing");
        assert!(preflight.plan.cancel.load(Ordering::SeqCst));
        assert!(preflight.task_result(done(&[])).is_err());
    }
}
