import os

import sys

from django.contrib.auth.models import User

from django.contrib.auth.forms import UserCreationForm

from django.contrib.auth import authenticate, login

from django.shortcuts import render, redirect

from .models import Agent, Environment, Policy

def index(request):

    if request.user.is_authenticated:

        agents = Agent.objects.all()

        environments = Environment.objects.all()

        policies = Policy.objects.all()

        return render(request, "index.html", {

            "agents": agents,

            "environments": environments,

            "policies": policies,

        })

    else:

        return redirect("login")

def create_agent(request):

    if request.user.is_authenticated:

        form = UserCreationForm(request.POST)

        if form.is_valid():

            new_user = form.save()

            agent = Agent.objects.create(

                user=new_user,

                policy=Policy.objects.get(name="Default"),

            )

            login(request, new_user)

            return redirect("index")

        else:

            return render(request, "create_agent.html", {

                "form": form,

            })

    else:

        return redirect("login")
def create_environment(request):

    if request.user.is_authenticated:

        form = EnvironmentForm(request.POST)

        if form.is_valid():

            new_environment = form.save()

            return redirect("index")

        else:

            return render(request, "create_environment.html", {

                "form": form,

            })

    else:

        return redirect("login")

def create_policy(request):

    if request.user.is_authenticated:

        form = PolicyForm(request.POST)

        if form.is_valid():

            new_policy = form.save()

            return redirect("index")

        else:

            return render(request, "create_policy.html", {

                "form": form,

            })

    else:

        return redirect("login")

def train_agent(request, agent_id):

    if request.user.is_authenticated:

        agent = Agent.objects.get(id=agent_id)

        environment = Environment.objects.get(id=agent.environment_id)

        policy = Policy.objects.get(id=agent.policy_id)

        for episode in range(1000):

            s = environment.reset()

            done = False

            episode_reward = 0

            while not done:

                a = policy(s)

                s_prime, r, done, _ = environment.step(a)

                episode_reward += r

                s = s_prime

            agent.update(episode_reward)

        return redirect("index")

    else:

        return redirect("login")
def test_agent(request, agent_id):

    if request.user.is_authenticated:

        agent = Agent.objects.get(id=agent_id)

        environment = Environment.objects.get(id=agent.environment_id)

        while True:

            s = environment.reset()

            done = False

            while not done:

                a = agent.policy(s)

                s_prime, r, done, _ = environment.step(a)

                environment.render()

                print("State:", s)

                print("Action:", a)

                print("Reward:", r)

                s = s_prime

                if done:

                    break

        return redirect("index")

    else:

        return redirect("login")

def login(request):

    if request.method == "POST":

        username = request.POST["username"]

        password = request.POST["password"]

        user = authenticate(username=username, password=password)

        if user is not None:

            login(request, user)

            return redirect("index")

        else:

            return render(request, "login.html", {

                "error_message": "Invalid username or password.",

            })
          def login(request):

    if request.method == "POST":

        username = request.POST["username"]

        password = request.POST["password"]

        user = authenticate(username=username, password=password)

        if user is not None:

            login(request, user)

            return redirect("index")

        else:

            return render(request, "login.html", {

                "error_message": "Invalid username or password.",

            })

    else:

        return render(request, "login.html")

def logout(request):

    logout(request)

    return redirect("login")

def create_user(request):

    if request.method == "POST":

        username = request.POST["username"]

        email = request.POST["email"]

        password = request.POST["password"]

        user = User.objects.create_user(

            username=username,

            email=email,

            password=password,

        )

        login(request, user)

        return redirect("index")

    else:

        return render(request, "create_user.html")

def change_password(request):

    if request.method == "POST":

        old_password = request.POST["old_password"]

        new_password = request.POST["new_password"]

        user = request.user

        if user.check_password(old_password):

            user.set_password(new_password)

            user.save()

            return redirect("index")

        else:
          return render(request, "change_password.html")
        def end_program(request):

    sys.exit()

def delete_user(request):

    if request.method == "POST":

        user = request.user

        user.delete()

        return redirect("index")

    else:

        return render(request, "delete_user.html")
