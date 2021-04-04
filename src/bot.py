from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3
from util.orientation import Orientation, relative_location

import math


class MyBot(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()
        self.bot_state = 0


    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """

        # Keep our boost pad info updated with which pads are currently active
        self.boost_pad_tracker.update_boost_status(packet)

        # This is good to keep at the beginning of get_output. It will allow you to continue
        # any sequences that you may have started during a previous call to get_output.
        if self.active_sequence is not None and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            if controls is not None:
                return controls

        # Gather some information about our car and the ball
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.game_ball.physics.location)
        ball_velocity = Vec3(packet.game_ball.physics.velocity)

        # By default we will chase the ball, but target_location can be changed later
        target_location = ball_location

        ball_prediction = self.get_ball_prediction_struct()  # This can predict bounces, etc
        # dynamic time to the ball depending on the car's speed
        time_in_future = self.time_to_ball(car_location, car_velocity.length(), ball_location)
        ball_in_future = find_slice_at_time(ball_prediction, packet.game_info.seconds_elapsed + time_in_future)

        # ball_in_future might be None if we don't have an adequate ball prediction right now, like during
        # replays, so check it to avoid errors.
        if ball_in_future is not None:
            target_location = Vec3(ball_in_future.physics.location)
            self.renderer.draw_line_3d(ball_location, target_location, self.renderer.cyan())

        # Draw some things to help understand what the bot is thinking
        self.renderer.draw_line_3d(car_location, target_location, self.renderer.white())
        self.renderer.draw_rect_3d(target_location, 8, 8, True, self.renderer.cyan(), centered=True)

        orientation = Orientation(my_car.physics.rotation)
        relative = relative_location(car_location, orientation, target_location)

        controls = SimpleControllerState()

        # jump if another car is close to not get stuck
        if self.location_to_nearest_car(car_location, my_car.team, packet) < 200 and car_velocity.length() < 50:
            controls.jump = True

        self.decide_state(controls, packet, my_car, car_location, car_velocity, ball_location, ball_velocity, target_location, ball_prediction, orientation, relative)

        return controls

    def decide_state(self, controls, packet, my_car, car_location, car_velocity, ball_location, ball_velocity, target_location, ball_prediction, orientation, relative):
        if self.bot_state == 0:
            self.ball_chase(controls, packet, my_car, car_velocity, car_location, target_location, ball_location, relative)
        elif self.bot_state == 1:
            self.retreat_to_goal(controls, packet, my_car, car_location)

    def ball_chase(self, controls, packet, my_car, car_velocity, car_location, target_location, ball_location, relative):
        self.renderer.draw_string_3d(car_location, 1, 1, "\nBall chasing", self.renderer.red())

        # retreat to own goal if the ball is a lot closer to our goal than we are
        info = self.get_field_info()
        own_goal_vec = info.goals[self.team].location
        own_goal_location = Vec3(own_goal_vec)
        enemy_goal_vec = info.goals[(self.team+1) % 2].location
        enemy_goal_location = Vec3(enemy_goal_vec)
        if ball_location.dist(own_goal_location) + 1000 < car_location.dist(own_goal_location):
            self.bot_state = 1

        self.renderer.draw_string_3d(ball_location, 1, 1, f"X: {ball_location.x} | Y {ball_location.y}", self.renderer.white())

        # makes the bots shoot towards the goal
        target_location = self.ball_towards_goal_location(target_location, own_goal_location, car_location, ball_location)
        self.renderer.draw_rect_3d(target_location, 8, 8, True, self.renderer.red(), centered=True)
        self.renderer.draw_line_3d(car_location, target_location, self.renderer.red())

        controls.steer = steer_toward_target(my_car, target_location)
        controls.throttle = 1.0
        # You can set more controls if you want, like controls.boost.

        # drift around
        if relative.x < -200:
            controls.handbrake = True
        elif 1000 < car_velocity.length() and abs(relative.y) < 15 and relative.x < 450 and relative.z < 100:
            # We'll do a front flip if the car is moving at a certain speed.
            return self.begin_front_flip(packet)

    def retreat_to_goal(self, controls, packet, my_car, car_location):
        self.renderer.draw_string_3d(car_location, 1, 1, "\nRetreating to goal", self.renderer.red())
        info = self.get_field_info()
        own_goal_vec = info.goals[self.team].location
        own_goal_location = Vec3(own_goal_vec)
        controls.steer = steer_toward_target(my_car, own_goal_location)
        controls.throttle = 1.0

        self.renderer.draw_string_3d(car_location, 1, 1, f"X: {own_goal_location.x} | Y {own_goal_location.y}", self.renderer.white())

        # change back to ball chasing if distance to goal is small
        if car_location.dist(own_goal_location) < 2000:
            self.bot_state = 0

    def go_towards_own_goal(self, controls, my_car, car_location, ball_location):
        self.renderer.draw_string_3d(car_location, 1, 1, "\nGoing towards own goal", self.renderer.red())
        info = self.get_field_info()
        own_goal_vec = info.goals[self.team].location
        own_goal_location = Vec3(own_goal_vec)
        controls.steer = steer_toward_target(my_car, own_goal_location)
        controls.throttle = 1.0

        # goes back to ball chase state if far enough away from the ball
        if car_location.dist(ball_location) > 1000 or car_location.dist(own_goal_location) < 2000:
            self.bot_state = 0

    def ball_towards_goal_location(self, target_location, goal_location, car_location, ball_location):
        slope = (goal_location.y - target_location.y) / (goal_location.x - target_location.x + 0.01)
        x_value = -1
        if target_location.x - goal_location.x < 0:
            x_value = 1
        """print("############")
        print("GOAL LOCATION")
        print(f"X: {goal_location.x}")
        print(f"Y: {goal_location.y}")
        print("############")
        print("Target LOCATION")
        print(f"X: {target_location.x}")
        print(f"Y: {target_location.y}")
        print("############")"""

        # angle from goal
        dist_from_x = abs(goal_location.x - target_location.x)

        self.renderer.draw_string_3d(car_location, 1, 1, f"\n\nDist {dist_from_x}", self.renderer.white())
        self.renderer.draw_string_3d(car_location, 1, 1, f"\n\n\nDist {target_location.y}", self.renderer.white())

        add = 0
        if target_location.y == 0:
            add = 1
        angle = 90 - abs(math.degrees(math.atan(dist_from_x / (target_location.y + add))))
        self.renderer.draw_string_3d(car_location, 1, 1, f"Angle {angle}", self.renderer.cyan())

        CORRECTION = 150
        new_target_location = Vec3(1, slope, 0).normalized()
        return target_location + x_value * CORRECTION*new_target_location



    def location_to_nearest_car(self, car_location, team, packet, enemy=False):
        # If enemy is true, only view nearest enemy cars
        nearest_distance = 999999
        for car in packet.game_cars:
            if car.team == team:
                continue
            other_car = Vec3(car.physics.location)
            distance_to = car_location.dist(other_car)
            if distance_to < nearest_distance:
                nearest_distance = distance_to
        return nearest_distance

    def time_to_ball(self, car_location, car_speed, ball_location):
        # estimates a time it takes for the bot to reach the ball for it to better predict where to go to hit the ball
        distance = car_location.dist(ball_location)
        return distance/(car_speed+0.01)

    def begin_front_flip(self, packet):
        # Send some quickchat just for fun
        self.send_quick_chat(team_only=False, quick_chat=QuickChatSelection.Information_IGotIt)

        # Do a front flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        self.active_sequence = Sequence([
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.1, controls=SimpleControllerState(jump=True, pitch=-1)),
            ControlStep(duration=0.8, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)
