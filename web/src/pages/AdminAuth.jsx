import {
  createContext, useCallback, useContext, useEffect, useState,
} from 'react';
import { api, setAuthToken, AUTH_LOST } from '../api/client.js';
import { Section, useToast, Empty } from '../components/ui.jsx';
import { IconLock, IconTrash, IconPlus } from '../components/Icons.jsx';

/**
 * The gate in front of the admin panel.
 *
 * Three states, and the third is the one that matters: a brand new install has
 * no account at all, so it has to let the first person in to create one --
 * otherwise there is no way to reach the screen that would let you in. The
 * server enforces the same rule, and bootstrap closes the door behind itself.
 */
export default function AdminGate({ children }) {
  const [state, setState] = useState({ status: 'loading' });

  const check = useCallback(() => {
    api.authStatus()
      .then((s) => setState({ status: 'ready', ...s }))
      .catch((e) => setState({ status: 'offline', error: e.message }));
  }, []);

  useEffect(() => { check(); }, [check]);

  // A token that stopped working anywhere in the app drops us back here.
  useEffect(() => {
    const onLost = () => check();
    window.addEventListener(AUTH_LOST, onLost);
    return () => window.removeEventListener(AUTH_LOST, onLost);
  }, [check]);

  if (state.status === 'loading') return null;

  if (state.status === 'offline') {
    return (
      <main className="page">
        <div className="page-inner">
          <Empty title="The API is not reachable" hint={state.error} />
        </div>
      </main>
    );
  }

  if (state.needsSetup) {
    return <SetupScreen onDone={check} />;
  }

  if (!state.user) {
    return <LoginScreen onDone={check} />;
  }

  return (
    <AuthContext.Provider value={{ user: state.user, refresh: check }}>
      {children}
    </AuthContext.Provider>
  );
}

/* --------------------------------------------------------------- context -- */

const AuthContext = createContext({ user: null, refresh: () => {} });
export const useAdminUser = () => useContext(AuthContext);

/* ---------------------------------------------------------------- screens -- */

function AuthShell({ title, hint, children, footer }) {
  return (
    <main className="page">
      <div className="auth-card">
        <div className="auth-mark"><IconLock /></div>
        <h1>{title}</h1>
        <p className="muted tiny">{hint}</p>
        {children}
        {footer}
      </div>
    </main>
  );
}

function SetupScreen({ onDone }) {
  const toast = useToast();
  const [form, setForm] = useState({ name: '', email: '', password: '', confirm: '' });
  const [busy, setBusy] = useState(false);
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  async function submit(e) {
    e.preventDefault();
    if (form.password.length < 8) return toast('Use at least 8 characters.', 'error');
    if (form.password !== form.confirm) return toast('The two passwords do not match.', 'error');
    setBusy(true);
    try {
      const res = await api.bootstrap({ email: form.email, password: form.password, name: form.name });
      setAuthToken(res.token);
      toast('Administrator account created.', 'ok');
      onDone();
    } catch (err) {
      toast(err.message, 'error');
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthShell
      title="Claim this installation"
      hint="The admin panel has no account yet, so anyone who reaches it can change the catalogue. Create the first administrator to close it."
    >
      <form onSubmit={submit}>
        <div className="field">
          <label>Your name</label>
          <input className="input" value={form.name} onChange={set('name')} autoComplete="name" />
        </div>
        <div className="field">
          <label>Email</label>
          <input className="input" type="email" required value={form.email}
                 onChange={set('email')} autoComplete="username" />
        </div>
        <div className="field">
          <label>Password</label>
          <input className="input" type="password" required minLength={8} value={form.password}
                 onChange={set('password')} autoComplete="new-password" />
          <span className="tiny dim">At least 8 characters.</span>
        </div>
        <div className="field">
          <label>Confirm password</label>
          <input className="input" type="password" required value={form.confirm}
                 onChange={set('confirm')} autoComplete="new-password" />
        </div>
        <button className="btn btn-primary" style={{ width: '100%' }} disabled={busy}>
          {busy ? 'Creating…' : 'Create administrator'}
        </button>
      </form>
      <p className="tiny dim" style={{ marginTop: 14 }}>
        On a server, set <code>ADMIN_EMAIL</code> and <code>ADMIN_PASSWORD</code> before
        the first start instead — then this screen never appears.
      </p>
    </AuthShell>
  );
}

function LoginScreen({ onDone }) {
  const toast = useToast();
  const [form, setForm] = useState({ email: '', password: '' });
  const [busy, setBusy] = useState(false);
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    try {
      const res = await api.login(form.email, form.password);
      setAuthToken(res.token);
      onDone();
    } catch (err) {
      toast(err.message, 'error');
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthShell title="Sign in" hint="The catalogue, rooms and enquiries are behind this.">
      <form onSubmit={submit}>
        <div className="field">
          <label>Email</label>
          <input className="input" type="email" required autoFocus value={form.email}
                 onChange={set('email')} autoComplete="username" />
        </div>
        <div className="field">
          <label>Password</label>
          <input className="input" type="password" required value={form.password}
                 onChange={set('password')} autoComplete="current-password" />
        </div>
        <button className="btn btn-primary" style={{ width: '100%' }} disabled={busy}>
          {busy ? 'Signing in…' : 'Sign in'}
        </button>
      </form>
    </AuthShell>
  );
}

/* ------------------------------------------------------------------ users -- */

export function UsersTab() {
  const toast = useToast();
  const { user, refresh } = useAdminUser();
  const [users, setUsers] = useState([]);
  const [form, setForm] = useState({ name: '', email: '', password: '', role: 'admin' });
  const [pw, setPw] = useState({ currentPassword: '', newPassword: '' });

  const reload = () => api.users().then(setUsers).catch((e) => toast(e.message, 'error'));
  useEffect(() => { reload(); }, []);

  async function add() {
    try {
      await api.createUser(form);
      setForm({ name: '', email: '', password: '', role: 'admin' });
      toast('Account created.', 'ok');
      reload();
    } catch (e) {
      toast(e.message, 'error');
    }
  }

  async function remove(u) {
    if (!confirm(`Remove ${u.email}? Their sessions end immediately.`)) return;
    try {
      await api.deleteUser(u.id);
      toast('Account removed.', 'ok');
      reload();
    } catch (e) {
      toast(e.message, 'error');
    }
  }

  async function changePassword() {
    try {
      await api.changePassword(pw.currentPassword, pw.newPassword);
      setPw({ currentPassword: '', newPassword: '' });
      toast('Password changed. Other sessions have been signed out.', 'ok');
    } catch (e) {
      toast(e.message, 'error');
    }
  }

  async function signOut() {
    await api.logout().catch(() => {});
    setAuthToken(null);
    refresh();
  }

  return (
    <>
      <div className="card">
        <div className="row-between">
          <div>
            <h3>Signed in as {user?.name || user?.email}</h3>
            <span className="tiny dim">{user?.email} · {user?.role}</span>
          </div>
          <button className="btn" onClick={signOut}>Sign out</button>
        </div>
      </div>

      <div className="grid-2" style={{ gap: 18, alignItems: 'start' }}>
        <div className="card">
          <h3>Add an account</h3>
          <div className="grid-2">
            <div className="field">
              <label>Name</label>
              <input className="input" value={form.name}
                     onChange={(e) => setForm({ ...form, name: e.target.value })} />
            </div>
            <div className="field">
              <label>Role</label>
              <select className="select" value={form.role}
                      onChange={(e) => setForm({ ...form, role: e.target.value })}>
                <option value="admin">Administrator</option>
                <option value="editor">Editor</option>
              </select>
            </div>
          </div>
          <div className="field">
            <label>Email</label>
            <input className="input" type="email" value={form.email}
                   onChange={(e) => setForm({ ...form, email: e.target.value })} />
          </div>
          <div className="field">
            <label>Password</label>
            <input className="input" type="password" value={form.password}
                   onChange={(e) => setForm({ ...form, password: e.target.value })} />
            <span className="tiny dim">At least 8 characters.</span>
          </div>
          <button className="btn btn-primary" onClick={add}><IconPlus /> Add account</button>
        </div>

        <div className="card">
          <h3>Change your password</h3>
          <div className="field">
            <label>Current password</label>
            <input className="input" type="password" value={pw.currentPassword}
                   onChange={(e) => setPw({ ...pw, currentPassword: e.target.value })}
                   autoComplete="current-password" />
          </div>
          <div className="field">
            <label>New password</label>
            <input className="input" type="password" value={pw.newPassword}
                   onChange={(e) => setPw({ ...pw, newPassword: e.target.value })}
                   autoComplete="new-password" />
          </div>
          <button className="btn btn-primary" onClick={changePassword}>Change password</button>
          <p className="tiny dim" style={{ marginTop: 8 }}>
            Every other session on this account is signed out when the password
            changes — which is what makes it useful after a laptop goes missing.
          </p>
        </div>
      </div>

      <Section title="Accounts">
        <table className="table">
          <thead>
            <tr><th>Name</th><th>Email</th><th>Role</th><th>Last signed in</th><th style={{ width: 50 }} /></tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id}>
                <td><strong>{u.name ?? '—'}</strong></td>
                <td className="muted">{u.email}</td>
                <td className="muted">{u.role}</td>
                <td className="muted tiny">{u.lastLogin ?? 'never'}</td>
                <td>
                  <button
                    className="btn btn-ghost btn-icon btn-sm"
                    disabled={u.id === user?.id}
                    title={u.id === user?.id ? 'This is you' : 'Remove'}
                    onClick={() => remove(u)}
                  >
                    <IconTrash />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>
    </>
  );
}
