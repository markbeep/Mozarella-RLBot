from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.ball_prediction_analysis import find_slice_at_time, find_matching_slice
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target, limit_to_safe_range
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
        ball_on_floor = target_location

        ball_prediction = self.get_ball_prediction_struct()  # This can predict bounces, etc
        # dynamic time to the ball depending on the car's speed
        time_in_future = self.time_to_ball(car_location, car_velocity.length(), ball_location)
        seconds_in_future = packet.game_info.seconds_elapsed + time_in_future
        ball_in_future = find_slice_at_time(ball_prediction, seconds_in_future)
        ball_on_floor = find_matching_slice(ball_prediction, 0, lambda s: s.physics.location.z < 150 and s.game_seconds >= packet.game_info.seconds_elapsed + time_in_future, search_increment=1)
        time_to_floor = 0

        # ball_in_future might be None if we don't have an adequate ball prediction right now, like during
        # replays, so check it to avoid errors.
        if ball_in_future is not None:
            target_location = Vec3(ball_in_future.physics.location)
            time_in_future = self.time_to_ball(car_location, car_velocity.length(), target_location)
            self.renderer.draw_line_3d(ball_location, target_location, self.renderer.cyan())

        # gets the next time when the ball is on the floor
        if ball_on_floor is not None:
            floor_location = Vec3(ball_on_floor.physics.location)
            time_to_floor = ball_on_floor.game_seconds - packet.game_info.seconds_elapsed
            self.renderer.draw_line_3d(ball_location, floor_location, self.renderer.orange())
            self.renderer.draw_rect_3d(floor_location, 8, 8, True, self.renderer.orange(), centered=True)

        # Draw some things to help understand what the bot is thinking
        self.renderer.draw_line_3d(car_location, target_location, self.renderer.white())
        self.renderer.draw_rect_3d(target_location, 8, 8, True, self.renderer.cyan(), centered=True)

        orientation = Orientation(my_car.physics.rotation)
        relative = relative_location(car_location, orientation, ball_location)

        controls = SimpleControllerState()

        # makes the car rotate to be more straight
        if not my_car.has_wheel_contact:
            # roll to land on all four wheels
            if orientation.roll < -0.1:
                controls.roll = 1
            elif orientation.roll > 0.1:
                controls.roll = -1

            # pitch to land on all four wheels
            if orientation.pitch < -0.1:
                controls.pitch = 1
            elif orientation.pitch > 0.1:
                controls.pitch = -1

            deg = math.degrees(car_location.ang_to(ball_location))

            # yaw to correct angle towards ball
            if deg < 85:
                controls.yaw = 1
            elif deg > 95:
                controls.yaw = -1

        # jump if another car is close to not get stuck
        if self.location_to_nearest_car(car_location, my_car.team, packet).dist(car_location) < 200 and car_velocity.length() < 50:
            controls.jump = True

        self.set_kickoff_state(car_velocity, ball_location, ball_velocity)

        self.decide_state(controls, packet, my_car, car_location, car_velocity, ball_location, ball_velocity, target_location, ball_prediction, orientation, relative, time_in_future, time_to_floor)

        return controls

    def decide_state(self, controls, packet, my_car, car_location, car_velocity, ball_location, ball_velocity, target_location, ball_prediction, orientation, relative, time_to_target, time_to_floor):
        if self.bot_state == 0:
            self.ball_chase(controls, packet, my_car, car_velocity, car_location, target_location, ball_location, relative, time_to_target, time_to_floor, orientation)
        elif self.bot_state == 1:
            self.retreat_to_goal(controls, packet, my_car, car_location, car_velocity)
        elif self.bot_state == 2:
            self.go_towards_own_goal(controls, my_car, car_location, ball_location)
        elif self.bot_state == 3:
            self.kickoff()

    def set_kickoff_state(self, car_velocity, ball_location, ball_velocity):
        if car_velocity.length() == 0 and ball_location.x == 0 and ball_location.y == 0 and ball_velocity.length() == 0:
            self.bot_state = 3

    def kickoff(self):
        self.bot_state = 0

    def ball_chase(self, controls, packet, my_car, car_velocity, car_location, target_location, ball_location, relative, time_to_target, time_to_floor, orientation):
        """
        Makes the bot chase the ball unless some conditions are valid
        """

        # retreat to own goal if the ball is a lot closer to our goal than we are
        info = self.get_field_info()
        own_goal_vec = info.goals[self.team].location
        own_goal_location = Vec3(own_goal_vec)

        if ball_location.dist(own_goal_location) + 1000 < car_location.dist(own_goal_location) and car_location.dist(own_goal_location) > 4000:
            self.bot_state = 1
            return
        elif own_goal_vec.y > 5000 and car_location.y + 100 < target_location.y:  # BLUE
            self.bot_state = 2
            return
        elif own_goal_vec.y < -5000 and car_location.y > target_location.y + 100:  # ORANGE
            self.bot_state = 2
            return

        self.renderer.draw_string_3d(car_location, 1, 1, "\nBall chasing", self.renderer.red())

        # makes the bots shoot towards the goal
        target_location = self.ball_towards_goal_location(target_location, own_goal_location, car_location, ball_location)
        self.renderer.draw_rect_3d(target_location, 8, 8, True, self.renderer.red(), centered=True)
        self.renderer.draw_line_3d(car_location, target_location, self.renderer.red())

        controls.steer = steer_toward_target(my_car, target_location)
        controls.throttle = 1.0
        # You can set more controls if you want, like controls.boost.

        # angle to ball
        car_to_ball = Vec3(ball_location.x - car_location.x, ball_location.y - car_location.y, ball_location.z - car_location.z)
        angle = math.degrees(orientation.forward.ang_to(car_to_ball))

        # boost
        if angle < 20 and not my_car.is_super_sonic:
            controls.boost = True

        # try to turn around quickly
        self.renderer.draw_string_2d(10, 10 + self.team * 100, 5, 5, f"{angle}", self.renderer.team_color())
        if angle > 160 and relative.x < -2000:
            self.begin_half_flip(packet)
        elif angle > 30:
            controls.handbreak = True
            controls.throttle = 0.7
            self.renderer.draw_string_3d(car_location, 1, 1, "\n\nDRIFTING", self.renderer.white())
        elif 1000 < car_velocity.length() and angle < 90 and car_to_ball.length() < 400 and relative.z < 200:
            # We'll do a front flip if the car is moving at a certain speed.
            return self.begin_front_flip(packet, angle, orientation.right.length())

    def drive_to_ball_bounce(self, my_car, car_location, floor_location):
        """
        Slowly drives to where the ball will bounce
        """
        pass

    def retreat_to_goal(self, controls, packet, my_car, car_location, car_velocity):
        """
        Makes the bot retreat back to the goal and only change back to another state when it's close to the goal
        """
        self.renderer.draw_string_3d(car_location, 1, 1, "\nRetreating to goal", self.renderer.red())
        info = self.get_field_info()
        own_goal_vec = info.goals[self.team].location
        own_goal_location = Vec3(own_goal_vec)
        controls.steer = steer_toward_target(my_car, own_goal_location)
        controls.throttle = 1.0

        if not my_car.is_super_sonic and car_velocity.length() > 200 and car_location.dist(own_goal_location) > 4500:
            controls.boost = True

        # change back to ball chasing if distance to goal is small
        self.renderer.draw_string_3d(car_location, 1, 1, f"\n\nDist to goal {car_location.dist(own_goal_location)}", self.renderer.white())
        if car_location.dist(own_goal_location) < 4000:
            self.bot_state = 0

    def go_towards_own_goal(self, controls, my_car, car_location, ball_location):
        """
        Goes towards own goal and changes back to ball chasing when a bit away from the ball
        """
        self.renderer.draw_string_3d(car_location, 1, 1, "\nGoing towards own goal", self.renderer.red())
        info = self.get_field_info()
        own_goal_vec = info.goals[self.team].location
        own_goal_location = Vec3(own_goal_vec)
        controls.steer = steer_toward_target(my_car, own_goal_location)
        controls.throttle = 1.0

        # goes back to ball chase state if far enough away from the ball
        if car_location.dist(ball_location) > 1000 or car_location.dist(own_goal_location) < 4000:
            self.bot_state = 0

    def jump_shot(self, controls, car_location, ball_location):
        pass

    def ball_towards_goal_location(self, target_location, goal_location, car_location, ball_location):
        """
        Modifies the target location so that the ball is hit a bit more towards the enemy goal
        """
        enemy_goal_location = Vec3(0, -goal_location.y, goal_location.z)

        if target_location.x - enemy_goal_location.x == 0:
            target_location.x = 1
            if goal_location.y < 0:  # usually for blue side
                target_location.x = -1
        slope = (target_location.y - enemy_goal_location.y) / (target_location.x - enemy_goal_location.x)

        dist = car_location.dist(target_location)
        if dist > 3000:
            correction = 1000
        elif 500 < dist <= 3000:
            correction = 0.36 * dist - 80
        else:
            correction = 100

        x_value = 1
        if target_location.x < 0:
            x_value = -1

        new_target_location = Vec3(1, slope, 0).normalized()
        return target_location + x_value * correction * new_target_location

    def location_to_nearest_car(self, car_location, team, packet, enemy=False):
        """
        Gets the closest enemy car to a target location
        """

        # If enemy is true, only view nearest enemy cars
        nearest_distance = 999999
        nearest_car = None
        for car in packet.game_cars:
            if car.team == team:
                continue
            other_car = Vec3(car.physics.location)
            distance_to = car_location.dist(other_car)
            if distance_to < nearest_distance:
                nearest_distance = distance_to
                nearest_car = other_car
        return nearest_car

    def time_to_ball(self, car_location, car_speed, ball_location):
        # estimates a time it takes for the bot to reach the ball for it to better predict where to go to hit the ball
        distance = car_location.dist(ball_location)
        return distance/(car_speed+100)

    def begin_half_flip(self, packet):
        self.active_sequence = Sequence([
            ControlStep(duration=1.0, controls=SimpleControllerState(throttle=-1, boost=False)),
            ControlStep(duration=0.1, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=1)),
            ControlStep(duration=0.15, controls=SimpleControllerState(pitch=-1, boost=False)),
            ControlStep(duration=0.5, controls=SimpleControllerState(pitch=-1, boost=True, roll=1, throttle=1)),
            ControlStep(duration=0.5, controls=SimpleControllerState()),
        ])

        return self.active_sequence.tick(packet)

    def begin_front_flip(self, packet, angle=0.0, right=1):
        # Do a flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        mult = 1
        if right < 0:
            mult = -1
        rad = math.radians(0)  # set to 0 for now
        print(rad)
        self.active_sequence = Sequence([
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.1, controls=SimpleControllerState(jump=True, pitch=-math.cos(rad), yaw=mult * math.sin(rad))),
            ControlStep(duration=0.8, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)
