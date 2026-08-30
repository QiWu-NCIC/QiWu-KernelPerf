const rows = document.getElementById("contestRows");
const empty = document.getElementById("contestEmpty");
const error = document.getElementById("contestError");

async function refreshContests() {
  try {
    const response = await fetch("/api/v1/contests");
    if (!response.ok) throw new Error(`Failed to load contests: ${response.status}`);
    const contests = await response.json();
    rows.replaceChildren();
    contests.forEach((contest) => {
      const row = document.createElement("tr");
      const idCell = document.createElement("td");
      const link = document.createElement("a");
      link.href = `/contest/${encodeURIComponent(contest.contest_id)}`;
      link.textContent = contest.contest_id;
      idCell.appendChild(link);
      const nameCell = document.createElement("td");
      nameCell.textContent = contest.name;
      const timeCell = document.createElement("td");
      timeCell.textContent = new Date(contest.start_time).toLocaleString();
      const createdCell = document.createElement("td");
      createdCell.textContent = new Date(contest.created_at).toLocaleString();
      row.append(idCell, nameCell, timeCell, createdCell);
      rows.appendChild(row);
    });
    empty.hidden = contests.length !== 0;
    error.hidden = true;
  } catch (cause) {
    error.textContent = String(cause);
    error.hidden = false;
  }
}

refreshContests();
